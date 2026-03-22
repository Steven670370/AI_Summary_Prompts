import numpy as np
from model import softmax, MiniTransformer
from AI_agent.memory import has_enough_data, get_high_quality_logs
from Transformer.tokenizer import tokenizer
from Transformer.model import save_model_weights

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

def train_one_step(model, input_ids, target_ids, lr=1e-3):
    logits = model.forward(np.array(input_ids))
    loss, probs = cross_entropy_loss(logits, target_ids)
    dlogits = cross_entropy_backward(probs, target_ids)
    model.backward(dlogits, lr)
    return loss

def update_knowledge(model, lr=1e-3):
    if not has_enough_data():
        print("no enough data")
        return 0

    training_data = get_high_quality_logs()
    learned = 0
    for q_text, r_text in training_data:
        q_ids = tokenizer.encode(q_text)
        r_ids = tokenizer.encode(r_text)

        min_len = min(len(q_ids), len(r_ids))
        if min_len == 0:
            continue
        loss = train_one_step(model, q_ids[:min_len], r_ids[:min_len], lr)
        learned += 1

    # save model
    save_model_weights(model)
    print(f"loss={loss:.4f}")
    return learned