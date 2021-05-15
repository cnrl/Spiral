import torch

def dense_connectivity(a_shape, b_shape):
    shape = (*a_shape, *b_shape)
    return torch.ones(shape, dtype=torch.bool)

def rfcpc_connectivity(a_shape, b_shape, c_count=None, c_rate=.1): #random fixed coupling prob connectivity
    shape = (*a_shape, *b_shape)
    rand_mat = torch.rand(shape)
    if c_count is None:
        c_count = int(rand_mat.numel() * c_rate)
    t = rand_mat.reshape(-1).sort()[0][-c_count]
    return (rand_mat >= t)

def rfnopp_connectivity(a_shape, b_shape, c_count=None, c_rate=.1): #random fixed number of presynaptic partners connectivity
    shape = (*a_shape, *b_shape)
    rand_mat = torch.rand(shape)
    flatted = rand_mat.reshape(-1, *b_shape)
    if c_count is None:
        c_count = int(flatted.shape[0] * c_rate)
    t = torch.topk(flatted, flatted.shape[0], dim=0, largest=False)[0][-c_count]
    return (rand_mat >= t)

def in_dense_connectivity(a_shape): # internal
    shape = (*a_shape, *a_shape)
    diag = torch.diag(torch.ones(torch.prod(torch.tensor(a_shape)))).reshape(shape)
    return (diag != 1)

def in_rfcpc_connectivity(a_shape, c_count=None, c_rate=.1): #internal random fixed coupling prob connectivity
    shape = (*a_shape, *a_shape)
    rand_mat = torch.rand(shape)
    diag = torch.diag(torch.ones(torch.prod(torch.tensor(a_shape)))).reshape(shape)
    diag = (diag==1)
    rand_mat[diag] = -1
    if c_count is None:
        c_count = int(rand_mat.numel() * c_rate)
    t = rand_mat.reshape(-1).sort()[0][-c_count]
    return (rand_mat >= t)

def in_rfnopp_connectivity(a_shape, c_count=None, c_rate=.1): #internal random fixed number of presynaptic partners connectivity
    shape = (*a_shape, *a_shape)
    rand_mat = torch.rand(shape)
    diag = torch.diag(torch.ones(torch.prod(torch.tensor(a_shape)))).reshape(shape)
    diag = (diag==1)
    rand_mat[diag] = -1
    flatted = rand_mat.reshape(-1, *a_shape)
    if c_count is None:
        c_count = int(flatted.shape[0] * c_rate)
    t = torch.topk(flatted, flatted.shape[0], dim=0, largest=False)[0][-c_count]
    return (rand_mat >= t)