import numpy as np

def cross_entropy(probs, labels, mean=False):
    if probs.shape != labels.shape:
        raise ValueError("Shape of probs and labels must match.")
    
    epsilon = 1e-15
    log_probs = np.log(probs + epsilon)
    loss = -np.sum(labels * log_probs, axis=1)
    if mean:
        loss = np.mean(loss)

    return loss

def softmax(logits):
    centered_logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(centered_logits)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return probs