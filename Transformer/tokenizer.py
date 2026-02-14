# tokenizer.py

class WordEntry:
    """
    Store one canonical word and its surface variants.
    """
    def __init__(self, word):
        self.base = word
        self.variants = set([word])

    def add_variant(self, word):
        self.variants.add(word)

class WordCollection:
    def __init__(self):
        # word → index
        self.word2index = {}
        # index → word
        self.index2word = []
        # word → WordEntry (for analysis only)
        self.entries = {}

    def _normalize(self, word):
        # 只做最小归一化
        return word.lower()

    def add_word(self, word):
        word = self._normalize(word)

        if word not in self.word2index:
            index = len(self.index2word)

            self.word2index[word] = index
            self.index2word.append(word)

            self.entries[word] = WordEntry(word)
        else:
            self.entries[word].add_variant(word)

    def encode(self, word):
        word = self._normalize(word)

        if word not in self.word2index:
            self.add_word(word)

        return self.word2index[word]

    def decode(self, index):
        return self.index2word[index]

    def vocab_size(self):
        return len(self.index2word)