import numpy as np

def generate(model, tokenizer, prompt, max_len=10):
    tokens = [tokenizer.encode(w) for w in prompt.split()]

    for _ in range(max_len):
        logits = model.forward(np.array(tokens))

        next_token = np.argmax(logits[-1])
        tokens.append(next_token)

    return " ".join([tokenizer.decode(t) for t in tokens])