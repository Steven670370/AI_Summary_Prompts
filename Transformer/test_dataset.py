# test_dataset.py

from dataset import TextDataset


class MockTokenizer:
    def __init__(self):
        self.vocab = {}

    def encode(self, word):
        if word not in self.vocab:
            self.vocab[word] = len(self.vocab) + 1
        return self.vocab[word]


def test_text_dataset_basic():
    dataset = TextDataset("hello world test", MockTokenizer(), seq_len=2)

    assert dataset.tokens == [1, 2, 3]

    assert dataset.inputs == [[1, 2]]
    assert dataset.targets == [[2, 3]]

    assert len(dataset) == 1

    assert dataset[0] == ([1, 2], [2, 3])