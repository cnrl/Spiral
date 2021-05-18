import torch

def constant_lr(lr):
    return lambda: torch.tensor(lr)

def constant_wdlr(lr): #weigth dependent learning rate
    return lambda w,wmin,wmax: torch.tensor(lr)

def stdp_wdlr(lr): #weigth dependent learning rate
    return lambda w,wmin,wmax: torch.tensor(lr) * (wmax-w) * (w-wmin)