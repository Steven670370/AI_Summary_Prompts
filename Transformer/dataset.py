# dataset.py

class TextDataset:
    def __init__(self, text, tokenizer, seq_len=4):
        """
        text: input text (string)
        tokenizer: WordCollection or similar
        seq_len: sequence length for Transformer input
        """
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        words = text.split()
        self.tokens = [tokenizer.encode(w) for w in words]

        self.inputs = []
        self.targets = []

        for i in range(len(self.tokens) - seq_len):
            input_seq = self.tokens[i : i + seq_len]
            target_seq = self.tokens[i + 1 : i + seq_len + 1]
            self.inputs.append(input_seq)
            self.targets.append(target_seq)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        # return a sequence
        return self.inputs[index], self.targets[index]