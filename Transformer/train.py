import numpy as np

from tokenizer import WordCollection
from dataset import TextDataset
from model import MiniTransformer, softmax
from loss import cross_entropy_loss, cross_entropy_backward

text = "hello world hello world hello world"

tokenizer = WordCollection()
dataset = TextDataset(text, tokenizer, seq_len=4)

model = MiniTransformer(
    vocab_size=tokenizer.vocab_size(),
    d_model=32,
    num_heads=4,
    d_ff=64,
    seq_len=4
)

lr = 1e-4

for epoch in range(200):

    total_loss = 0

    for i in range(len(dataset)):

        x_ids, y_ids = dataset[i]

        x_ids = np.array(x_ids)
        y_ids = np.array(y_ids)

        # -------- forward --------
        logits = model.forward(x_ids)

        # -------- loss --------
        loss, probs = cross_entropy_loss(logits, y_ids)
        total_loss += loss

        # -------- backward --------
        dlogits = cross_entropy_backward(probs, y_ids)
        model.backward(dlogits, lr)

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")


def predict(model, tokenizer, start_words, steps=5):
    tokens = [tokenizer.encode(w) for w in start_words]

    for _ in range(steps):
        x = np.array(tokens[-4:])
        logits = model.forward(x)
        probs = softmax(logits[-1])
        next_token = np.argmax(probs)
        tokens.append(next_token)

    return [tokenizer.decode(t) for t in tokens]