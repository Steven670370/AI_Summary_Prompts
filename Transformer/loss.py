import numpy as np
from model import softmax

def cross_entropy_loss(logits, targets):
    probs = softmax(logits)
    correct = probs[np.arange(len(targets)), targets]
    loss = -np.mean(np.log(correct + 1e-9))
    return loss, probs

def cross_entropy_backward(probs, targets):
    dlogits = probs.copy()
    dlogits[np.arange(len(targets)), targets] -= 1
    dlogits /= len(targets)
    return dlogits