from playgroundrl.client import *
from playgroundrl.actions import *
import gensim
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet

google_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, limit=500000)

BOARD_SIZE = 25


class TestCodenames(PlaygroundClient):
    def __init__(self, render: bool):
        self.guessed_words = set()
        self.gone_through = 0
        self.threshold = 0
        super().__init__(
            GameType.CODENAMES,
            model_name="ashley-bot-wordnet",
            auth={
                "email": "daia12@stanford.edu",
                "api_key": "pe0YAsFDGcrGtqU-WVuMpRgh8VN9w8Ue_g1L9mBtm5k",
            },
            render_gameplay=render,
        )

    # ----------------- HELPERS STARTING HERE --------------------------
    def find_additional_words(self, state: CodenamesState, clueword):
      count = 1
      for i in range(BOARD_SIZE):
        if state.guessed[i] == "UNKNOWN" and state.actual[i] == state.color:
          if len(wordnet.synsets(state.words[i])) != 0:
            curr_syn = wordnet.synsets(state.words[i])[0]
            if wordnet.path_similarity(curr_syn, clueword) > self.threshold:
              count += 1
      return count

    def get_similarity_list(self, state: CodenamesState):
      sim_list =[]
      clue_syn = wordnet.synsets(state.clue)[0]
      for i in range(BOARD_SIZE):
        if state.guessed[i] == "UNKNOWN":
          if len(wordnet.synsets(state.words[i])) != 0:
            curr_syn = wordnet.synsets(state.words[i])[0]
            sim_list.append((i, wordnet.path_similarity(curr_syn, clue_syn)))
          else: 
             sim_list.append((i, 0))
      return sorted(sim_list, key = lambda x: x[1], reverse=True)

    # warning: this could increase runtime
    def check_validity(self, state: CodenamesState, pot_word, pot_syn, target_syn, assasyn):
      if pot_word.isalpha():
        for i in range(BOARD_SIZE):
          if pot_word in state.words[i] or state.words[i] in pot_word:
            return False
        # making sure our clueword is not more similar to the bomb word
        if wordnet.path_similarity(pot_syn, target_syn) > wordnet.path_similarity(pot_syn, assasyn):
          return True
      return False

    def train(self):
      new_model = google_model
      return new_model


    # this function picks words of interest in a distributed manor to conserve runtime
    # and improve game play
    def find_effective_word(self, state: CodenamesState):
      # wrap around
      starting_point = (self.gone_through + 1) % BOARD_SIZE
      for i in range(starting_point, BOARD_SIZE):
        if state.guessed[i] == "UNKNOWN" and state.actual[i] == state.color:
          self.gone_through = i
          return state.words[i]
      for j in range(starting_point):
        if state.guessed[j] == "UNKNOWN" and state.actual[j] == state.color:
          self.gone_through = j
          return state.words[j]

    def find_black_word(self, state: CodenamesState):
      for i in range(BOARD_SIZE):
        if state.actual[i] == "ASSASSIN":
          return state.words[i]
    
    def most_similar(self, state: CodenamesState, model, woi, antiword, target_syn, assasyn):
      if antiword in model.key_to_index:
        candidate = model.most_similar(positive=[woi], negative=[antiword], topn=20)
      else:
        candidate = model.most_similar(positive=[woi], topn=20)
      valid = False
      clueword = 'clue'
      i = 0
      while i < 20 and not valid:
        potential = candidate[i][0].lower()
        if potential in self.guessed_words:
          print(potential + " already guessed!")
        else:
          if len(wordnet.synsets(potential)) != 0:
            if self.check_validity(state, potential, wordnet.synsets(potential)[0], target_syn, assasyn):
              clueword = potential
              valid = True
        i+=1
      return clueword

    # -------------------- HELPERS END HERE ------------------------

    def callback(self, state: CodenamesState, reward):
        new_model = self.train()
        if state.player_moving_id not in self.player_ids:
            return None

        if state.role == "GIVER":
            print('Giving....')
            print('Getting effective_word...')
            word_of_interest = self.find_effective_word(state)
            print('Effective words is ' + word_of_interest)
            antiword = self.find_black_word(state)
            clueword = 'clue'
            print('Assassin is ' + antiword)
            print('Finding best clueword...')
            # ------ get synsets ----
            syn = wordnet.synsets(word_of_interest)
            if (wordnet.synsets(antiword)) != 0:
              assasyn  = wordnet.synsets(antiword)[0]
            else:
              assasyn  = wordnet.synsets('clue')[0]
            target_syn = syn[0]
            

            # order of presedence:
            # first holonyms and hypernyms 
            candidates = set()
            picked_word = False
            for lemma in syn[0].part_holonyms():
              candidates.add(lemma)
            for lemma in syn[0].hypernyms():
              candidates.add(lemma)

            for candidate in candidates:
              name = candidate.name().split('.')[0]
              if name in self.guessed_words:
                print(name + " already guessed!")
              elif self.check_validity(state, name, candidate, target_syn, assasyn):
                  self.threshold = (wordnet.path_similarity(candidate, assasyn) + wordnet.path_similarity(candidate, target_syn))/2
                  clueword = name
                  self.guessed_words.add(clueword)
                  count = self.find_additional_words(state, candidate)
                  picked_word = True
            
            # then synonyms
            if not picked_word:
              candidates = set()
              for lemma in syn:
                candidates.add(lemma)

              for candidate in candidates:
                name = candidate.name().split('.')[0]
                if name in self.guessed_words:
                  print(name + " already guessed!")
                elif self.check_validity(state, name, candidate, target_syn, assasyn):
                    self.threshold = (wordnet.path_similarity(candidate, assasyn) + wordnet.path_similarity(candidate, target_syn))/2
                    clueword = name
                    count = self.find_additional_words(state, candidate)
                    self.guessed_words.add(clueword)
                    picked_word = True

            # then related words
            if not picked_word:
              for lemma in syn[0].also_sees() :
                name = lemma.name().split('.')[0]
                if name in self.guessed_words:
                  print(name + " already guessed!")
                elif self.check_validity(state, name, lemma, target_syn, assasyn):
                    self.threshold = (wordnet.path_similarity(lemma, assasyn) + wordnet.path_similarity(lemma, target_syn))/2
                    clueword = name
                    count = self.find_additional_words(state, lemma)
                    self.guessed_words.add(clueword)
                    
            # then similar words 
            if not picked_word:
              for lemma in syn[0].similar_tos():
                name = lemma.name().split('.')[0]
                if name in self.guessed_words:
                  print(name + " already guessed!")
                elif self.check_validity(state, name, lemma, target_syn, assasyn):
                    self.threshold = (wordnet.path_similarity(lemma, assasyn) + wordnet.path_similarity(lemma, target_syn))/2
                    clueword = name
                    count = self.find_additional_words(state, lemma)
                    self.guessed_words.add(clueword)

            # fall back on old model
            if not picked_word:
              clueword = self.most_similar(state, new_model, word_of_interest, antiword, target_syn, assasyn)
              lemma = wordnet.synsets(clueword)[0]
              self.threshold = (wordnet.path_similarity(lemma, assasyn) + wordnet.path_similarity(lemma, target_syn))/2
              count = self.find_additional_words(state, lemma)
              self.guessed_words.add(clueword)

            print("clueword is " + clueword)

            return CodenamesSpymasterAction(
                word = clueword,
                count = 1
            )
        elif state.role == "GUESSER":
          print('Guessing...')
          sim_list = self.get_similarity_list(state)
          answer_list = [i[0] for i in sim_list]
          if state.count > len(sim_list):
            return CodenamesGuesserAction(
            guesses = answer_list[:1]
            # guesses = [self.get_most_similar(state, new_model)],
            # guesses = self.get_my_squares(state)[:2],
          )
          else:
            return CodenamesGuesserAction(
                guesses = answer_list[:1]
                # guesses = [self.get_most_similar(state, new_model)],
            )

    def gameover_callback(self):
        print('gameover')
        pass

if __name__ == "__main__":
    t = TestCodenames(True)
    print('running...')
    t.run(
        num_games = 10,
        pool=Pool(Pool.OPEN),
        # num_games=args.num_games,
        self_training= False,
        maximum_messages=500000,
    )
