import torch
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

def mae(outputs, targets, reduction='mean', **kwargs):
    return F.l1_loss(outputs, targets, reduction=reduction)

def rmse(outputs, targets, reduction='mean', **kwargs):
    return torch.sqrt(F.mse_loss(outputs, targets, reduction=reduction))

def _find_repeats(data):
    temp = data.detach().clone()
    temp = temp.sort()[0]
    change = torch.cat([torch.tensor([True], device=temp.device), temp[1:] != temp [:-1]])
    unique = temp[change]
    change_idx = torch.cat([torch.nonzero(change), torch.tensor([[temp.numel()]], device=temp.device)]).flatten()
    freq = change_idx[1:] - change_idx[:-1]
    atleast2 = freq > 1
    return unique[atleast2]

def _rank_data(data):
    n = data.numel()
    rank = torch.empty_like(data)
    idx = data.argsort()
    rank[idx[:n]] = torch.arange(1, n+1, dtype=data.dtype, device=data.device)
    
    repeats = _find_repeats(data)
    for r in repeats:
        condition = data == r
        rank[condition] = rank[condition].mean()
    return rank
    
def spearmanr(preds, target, eps=1e-6, **kwargs):
    preds = _rank_data(preds)
    target = _rank_data(target)
    
    preds_diff = preds - preds.mean()
    target_diff = target - target.mean()
    
    cov = (preds_diff * target_diff).mean()
    preds_std = torch.sqrt((preds_diff * preds_diff).mean())
    target_std = torch.sqrt((target_diff * target_diff).mean())
    
    corrcoef = cov / (preds_std * target_std + eps)
    return torch.clamp(corrcoef, -1.0, 1.0)

def pearsonr(preds, target, eps=1e-6, **kwargs):
    mean_x = preds.mean()
    mean_y = target.mean()
    var_x = ((preds - mean_x) * (preds - mean_x)).sum()
    var_y = ((target - mean_y) * (target - mean_y)).sum()
    corr_xy = ((preds - mean_x) * (target - mean_y)).sum()
    corrcoef = (corr_xy / (var_x * var_y + eps).sqrt())
    return torch.clamp(corrcoef, -1.0, 1.0)

def qwk(preds, target, prompt, eps=1e-6):
    
    n_classes = {
        1: 11,
        2: 6,
        3: 4,
        4: 4,
        5: 5,
        6: 5,
        7: 31,
        8: 61,
    }
    
    _preds = F.one_hot((preds * (n_classes[prompt]-1)).long(), num_classes=n_classes[prompt]).float()
    _target = F.one_hot((target * (n_classes[prompt]-1)).long(), num_classes=n_classes[prompt]).float()
    
    weights = torch.arange(0, n_classes[prompt], dtype=torch.float32, device=preds.device) / (n_classes[prompt] - 1)
    weights = (weights - torch.unsqueeze(weights, -1)) ** 2
    
    O = torch.matmul(_target.t(), _preds)
    E = torch.matmul(_target.sum(dim=0).view(-1,1), _preds.sum(dim=0).view(1,-1)) / O.sum()
    
    return 1 - (weights*O).sum() / ((weights*E).sum() + eps)

def acc(preds, target):
    preds = preds > 0.5
    return accuracy_score(target.int(), preds.int())

def cf_matrix(preds, target):
    preds = preds > 0.5
    return confusion_matrix(target.int(), preds.int())

def f1(preds, target):
    preds = preds > 0.5
    return f1_score(target.int(), preds.int(), average='binary')

def prf(preds, targets):
    preds = preds > 0.5
    p, r, f, _ = precision_recall_fscore_support(targets.int(), preds.int(), average='binary')
    return p, r, f
    
METRICS = {
    'rmse': rmse,
    'mae': mae,
    'qwk': qwk,
    'spr': spearmanr,
    'prs': pearsonr,
    'acc': acc,
    'f1': f1,
    'prf': prf,
}
    
    
    
    