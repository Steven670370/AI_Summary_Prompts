# dataset.py

class TextDataset:
    def __init__(self, text, tokenizer):
        """
        text: input text
        tokenizer: WordCollection
        """
        self.tokenizer = tokenizer

        # divide into words
        words = text.split()

        # use the set to contain their number
        self.tokens = [tokenizer.encode(w) for w in words]

        # predict the next output based on the previous inputs
        self.inputs = self.tokens[:-1]
        self.targets = self.tokens[1:]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]