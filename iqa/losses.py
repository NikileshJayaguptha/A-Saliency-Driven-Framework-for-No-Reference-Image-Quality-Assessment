import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr


def plcc_loss(pred, target, eps=1e-8):
    pred, target = pred.view(-1), target.view(-1)
    pc, tc = pred - pred.mean(), target - target.mean()
    plcc = (pc * tc).sum() / (torch.sqrt((pc ** 2).sum() * (tc ** 2).sum()) + eps)
    return 1 - plcc


def rank_loss(pred, target):
    pred, target = pred.view(-1), target.view(-1)
    diff_pred = pred.unsqueeze(0) - pred.unsqueeze(1)
    sign_target = torch.sign(target.unsqueeze(0) - target.unsqueeze(1))
    return F.relu(-diff_pred * sign_target).mean()


def combined_loss(pred, target, w_plcc=0.4, w_rank=0.1):
    return F.mse_loss(pred.view(-1), target.view(-1)) + w_plcc * plcc_loss(pred, target) + w_rank * rank_loss(pred, target)


def metrics(preds, labels):
    preds, labels = np.asarray(preds, dtype=np.float64), np.asarray(labels, dtype=np.float64)
    return {
        "PLCC": float(pearsonr(preds, labels)[0]),
        "SRCC": float(spearmanr(preds, labels)[0]),
        "RMSE": float(np.sqrt(np.mean((preds - labels) ** 2))),
    }
