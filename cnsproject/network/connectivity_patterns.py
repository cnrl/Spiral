import torch

def dense_connectivity(a_shape, b_shape):
    shape = (a_shape, b_shape)
    return torch.ones(shape, dtype=torch.bool)

def rfcpc_connectivity(a_shape, b_shape, c_count=None, c_rate=.1): #random fixed coupling prob connectivity
    shape = (a_shape, b_shape)
    rand_mat = torch.rand(shape)
    if c_count is None:
        c_count = int(rand_mat.numel() * c_rate)
    t = rand_mat.reshape(-1).sort()[0][-c_count]
    return (rand_mat >= t)

def rfnopp_connectivity(a_shape, b_shape, c_count=None, c_rate=.1): #random fixed number of presynaptic partners connectivity
    shape = (a_shape, b_shape)
    rand_mat = torch.rand(shape)
    flatted = rand_mat.reshape(-1, *postshape)
    if c_count is None:
        c_count = int(flatted.shape[0] * c_rate)
    t = torch.topk(flatted, flatted.shape[0], dim=0, largest=False)[0][-c_count]
    return (rand_mat >= t)

def internal_dense_connectivity(a_shape):
    shape = (a_shape, a_shape)
    diag = torch.diag(torch.ones(preshape)).reshape(shape)
    return (diag != 1)

def internal_rfcpc_connectivity(a_shape, c_count=None, c_rate=.1): #random fixed coupling prob connectivity
    shape = (a_shape, a_shape)
    rand_mat = torch.rand(shape)
    diag = torch.diag(torch.ones(connection.pre)).reshape(shape)
    diag = (diag==1)
    rand_mat[diag] = -1
    if c_count is None:
        c_count = int(rand_mat.numel() * c_rate)
    t = rand_mat.reshape(-1).sort()[0][-c_count]
    return (rand_mat >= t)

def internal_rfnopp_connectivity(a_shape, c_count=None, c_rate=.1): #random fixed number of presynaptic partners connectivity
    shape = (a_shape, a_shape)
    rand_mat = torch.rand(shape)
    diag = torch.diag(torch.ones(connection.pre)).reshape(shape)
    diag = (diag==1)
    rand_mat[diag] = -1
    flatted = rand_mat.reshape(-1, a_shape)
    c_count = kwargs.get("c_count", None)
    if c_count is None:
        c_count = int(flatted.shape[0] * c_rate)
    t = torch.topk(flatted, flatted.shape[0], dim=0, largest=False)[0][-c_count]
    return (rand_mat >= t)