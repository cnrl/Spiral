import torch

def constant_initialization(a_shape, b_shape, w_scale=1):
    shape = (a_shape, b_shape)
    return w_scale*torch.ones(shape)

def uniform_initialization(a_shape, b_shape, w_range=(0,1)):
    shape = (a_shape, b_shape)
    wmin = w_range[0]
    wmax = w_range[1]
    return torch.rand(shape)*(wmax-wmin)+wmin

def norm_initialization(a_shape, b_shape, w_mean=1., w_std=.1):
    shape = (a_shape, b_shape)
    w = torch.normal(w_mean, w_std, shape)
    return w