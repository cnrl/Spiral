import torch

# abstract function!!
def dense_connectivity(preshape, postshape, **kwargs):
    shape = (*preshape, *postshape)
    return torch.ones(shape, dtype=torch.bool)

def rfcpc_connectivity(preshape, postshape, **kwargs): #random fixed coupling prob connectivity
    shape = (*preshape, *postshape)
    rand_mat = torch.rand(shape)
    c = kwargs.get("connections_count", None)
    if c is None:
        p = kwargs.get("connections_rate", .1)
        c = int(rand_mat.numel() * p)
    t = rand_mat.reshape(-1).sort()[0][-c]
    return (rand_mat >= t)

def rfnopp_connectivity(preshape, postshape, **kwargs): #random fixed number of presynaptic partners connectivity
    shape = (*preshape, *postshape)
    rand_mat = torch.rand(shape)
    flatted = rand_mat.reshape(-1, *postshape)
    c = kwargs.get("connections_count", None)
    if c is None:
        p = kwargs.get("connections_rate", .1)
        c = int(flatted.shape[0] * p)
    t = torch.topk(flatted, flatted.shape[0], dim=0, largest=False)[0][-c]
    return (rand_mat >= t)

def internal_dense_connectivity(preshape, postshape, **kwargs):
    diag = torch.diag(torch.ones(preshape)).reshape(*preshape,*preshape)
    return (diag != 1)

def internal_rfcpc_connectivity(preshape, postshape, **kwargs): #random fixed coupling prob connectivity
    shape = (*preshape, *preshape)
    rand_mat = torch.rand(shape)
    diag = torch.diag(torch.ones(preshape)).reshape(*preshape,*preshape)
    diag = (diag==1)
    rand_mat[diag] = -1
    c = kwargs.get("connections_count", None)
    if c is None:
        p = kwargs.get("connections_rate", .1)
        c = int(rand_mat.numel() * p)
    t = rand_mat.reshape(-1).sort()[0][-c]
    return (rand_mat >= t)

def internal_rfnopp_connectivity(preshape, postshape, **kwargs): #random fixed number of presynaptic partners connectivity
    shape = (*preshape, *preshape)
    rand_mat = torch.rand(shape)
    diag = torch.diag(torch.ones(preshape)).reshape(*preshape,*preshape)
    diag = (diag==1)
    rand_mat[diag] = -1
    flatted = rand_mat.reshape(-1, *preshape)
    c = kwargs.get("connections_count", None)
    if c is None:
        p = kwargs.get("connections_rate", .1)
        c = int(flatted.shape[0] * p)
    t = torch.topk(flatted, flatted.shape[0], dim=0, largest=False)[0][-c]
    return (rand_mat >= t)


# abstract function!!
def constant_weights(preshape, postshape, **kwargs):
    shape = (*preshape, *postshape)
    return kwargs.get('wscale',1)*torch.ones(shape)

def uniform_weights(preshape, postshape, **kwargs):
    shape = (*preshape, *postshape)
    wmin = kwargs.get('wmin',0)
    wmax = kwargs.get('wmax',1)
    return torch.rand(shape)*(wmax-wmin)+wmin

def norm_weights(preshape, postshape, **kwargs):
    shape = (*preshape, *postshape)
    return torch.normal(kwargs.get('wmean',1.), kwargs.get('wstd',.1), shape)