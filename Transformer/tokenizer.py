# tokenizer.py

class WordEntry:
    """
    - base: stem of words
    - variants: original words
    """
    def __init__(self, base_word, original_word):
        self.base = base_word
        self.variants = set([original_word])

    def add_variant(self, word):
        self.variants.add(word)

class WordCollection:
    """
    A collection of all words
    """
    def __init__(self):
        # key: base word, value: WordEntry
        self.words = {}


    def _normalize(self, word):
        """
        Removing prefixes and suffixes, such as -ing, -ed, -s, -es
        """
        word = word.lower()

        for prefix in ['un', 're', 'pre']:
            if word.startswith(prefix):
                word = word[len(prefix):]

        for suffix in ['ing', 'ed', 'ly', 'es', 's']:
            if word.endswith(suffix):
                word = word[:-len(suffix)]

        if len(word) >= 2 and word[-1] == word[-2] and word[-1] not in 'aeiou':
            word = word[:-1]

        if word.endswith('i'):
            word = word[:-1] + 'y'

        return word
    

    def add_word(self, word):
        base = self._normalize(word)
        if base in self.words:
            self.words[base].add_variant(word)
        else:
            self.words[base] = WordEntry(base, word)

    
    def encode(self, word):
        """
        return index
        """
        base = self._normalize(word)
        if base not in self.words:
            self.add_word(word)

        return list(sorted(self.words.keys())).index(base)


    def decode(self, index):
        """
        return base word
        """
        base = list(sorted(self.words.keys()))[index]
        return base