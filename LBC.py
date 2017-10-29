from utils import tokenize


class LBC:

    booster_words = [
            'muito',
            'super',
    ]

    downtoner_words = [
            'pouco',
            'apenas'
    ]

    negative_words = [
            'não',
            'nem'
    ]

    BOOST_POLARITY = 3

    def __init__(self, sentiment_lexicon_file):
        self.sentiment_lexicon = self.read_lexicon(
                self.liwc_reader, sentiment_lexicon_file)

    def read_lexicon(self, reader, lexicon_file):
        return reader(lexicon_file)

    def liwc_reader(self, lexicon_file):
        sentiment_lex = {}
        posemo = '126'
        negemo = '127'
        words_tags = open(lexicon_file, 'r').readlines()[66:]
        for entry in words_tags:
            w_tags = entry.split('\t')
            word = w_tags[0]
            if word[-1] != '*' \
                    and word not in self.booster_words \
                    and word not in self.downtoner_words \
                    and word not in self.negative_words:
                if posemo in w_tags:
                    sentiment_lex[word] = 1
                elif negemo in w_tags:
                    sentiment_lex[word] = -1
        return sentiment_lex

    def get_polarity(self, word):
        if word in self.sentiment_lexicon:
            return self.sentiment_lexicon[word]
        return 0

    def context_polarity(self, tokens, sent_word_idx):
        negation = False
        booster = False
        downtoner = False
        sentiment_word = tokens[sent_word_idx]
        word_polarity = self.get_polarity(sentiment_word)
        if len(list(set(tokens[:sent_word_idx]) &
                    set(self.negative_words))) > 0:
            negation = True
        if len(list(set(tokens[:sent_word_idx]) &
                    set(self.booster_words))) > 0:
            booster = True
        if len(list(set(tokens[:sent_word_idx]) &
                    set(self.downtoner_words))) > 0:
            downtoner = True
        if negation:
            if downtoner:
                return self.BOOST_POLARITY * word_polarity
            if booster:
                return 1 / self.BOOST_POLARITY * word_polarity
            return -1 * word_polarity
        elif booster:
            return self.BOOST_POLARITY * word_polarity
        elif downtoner:
            return 1 / self.BOOST_POLARITY * word_polarity
        return word_polarity

    def classify(self, text):
        tokens = tokenize(text)
        return sum([self.context_polarity(tokens, idx)
                    for idx in range(len(tokens))])


if __name__ == '__main__':
    classifier = LBC('./data/LIWC2007_Portugues_win.dic.txt')
    text = 'não é legal o filme'
    result = classifier.classify(text)
    print(text + '\t' + str(result))
