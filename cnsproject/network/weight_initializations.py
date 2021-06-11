import torch

def lazy_initialization(data):
    return lambda a_shape, b_shape:  data

def constant_initialization(w_scale=1):
    return lambda a_shape, b_shape:  w_scale*torch.ones((*a_shape, *b_shape))

def uniform_initialization(w_range=(0,1)):
    wmin = w_range[0]
    wmax = w_range[1]
    return lambda a_shape, b_shape: torch.rand((*a_shape, *b_shape))*(wmax-wmin)+wmin

def norm_initialization(w_mean=.5, w_std=.1):
    return lambda a_shape, b_shape: torch.normal(w_mean, w_std, (*a_shape, *b_shape))