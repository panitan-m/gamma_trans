import math

import torch
import torch.nn.functional as F

def mse(y_pred, y_true, **kwargs):
    return F.mse_loss(y_pred, y_true)


def listNet(y_pred, y_true, eps=1e-10, **kwargs):
    
    preds_smax = F.softmax(y_pred, 0)
    true_smax = F.softmax(y_true, 0)
    
    preds_smax = preds_smax + eps
    preds_log = torch.log(preds_smax)

    return torch.mean(-torch.sum(true_smax * preds_log, 0))


def r2(y_pred, y_true, epochs, epoch, gamma=0.99999, eps=1e-10):
    
    Lm = LOSS_FN['reg'](y_pred, y_true)
    Lr = LOSS_FN['rank'](y_pred, y_true, eps=eps)
    
    if epochs is not None and epoch is not None:
        Te = 1 / (1 + math.exp(gamma * (epochs / 2 - epoch)))
    else:
        Te = 1
    
    return Te * Lm + (1 - Te) * Lr

def bce(y_pred, y_true, **kwargs):
    return F.binary_cross_entropy(y_pred, y_true)

LOSS_FN = {
    'reg': mse,
    'rank': listNet,
    'r2': r2,
    'bce': bce,
}