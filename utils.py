import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import config

def compute_auroc(outputs, targets):
    """
    outputs: tensor of shape (N,14) containing model logits/probs
    targets: tensor of shape (N,14) containing binary targets
    """
    probs = torch.sigmoid(outputs).cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()
    aurocs = []
    for i in range(config.NUM_CLASSES):
        try:
            score = roc_auc_score(targets[:, i], probs[:, i])
            aurocs.append(score)
        except ValueError: pass
    return 0.0 if len(aurocs) == 0 else np.mean(aurocs)
