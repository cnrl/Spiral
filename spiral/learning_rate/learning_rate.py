import torch

def constant_lr(lr):
    return lambda: torch.as_tensor(lr)

def constant_wdlr(lr): #weigth dependent learning rate
    return lambda w,wmin,wmax: torch.as_tensor(lr)

def stdp_wdlr(lr): #weigth dependent learning rate
    return lambda w,wmin,wmax: torch.as_tensor(lr) * (wmax-w) * (w-wmin)