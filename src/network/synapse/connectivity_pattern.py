import torch

def dense_connectivity(internal=False):
    if not internal:
        return lambda a,b: torch.ones((*a, *b), dtype=torch.bool)
    else:
        def func(a_shape, _):
            shape = (*a_shape, *a_shape)
            diag = torch.diag(torch.ones(torch.prod(torch.tensor(a_shape)))).reshape(shape)
            return (diag != 1)
        return func

def rfccc_connectivity(c_count, internal=False): #random fixed coupling count connectivity
    def func(a_shape, b_shape):
        shape = (*a_shape, *b_shape)
        rand_mat = torch.rand(shape)
        if internal:
            diag = torch.diag(torch.ones(torch.prod(torch.tensor(a_shape)))).reshape(shape)
            diag = (diag==1)
            rand_mat[diag] = -1
        t = rand_mat.reshape(-1).sort()[0][-c_count]
        return (rand_mat >= t)
    return func

def rfnopp_connectivity(c_count, internal=False): #random fixed number of presynaptic partners connectivity
    def func(a_shape, b_shape):
        shape = (*a_shape, *b_shape)
        rand_mat = torch.rand(shape)
        if internal:
            diag = torch.diag(torch.ones(torch.prod(torch.tensor(a_shape)))).reshape(shape)
            diag = (diag==1)
            rand_mat[diag] = -1
        flatted = rand_mat.reshape(-1, *b_shape)
        t = torch.topk(flatted, flatted.shape[0], dim=0, largest=False)[0][-c_count]
        return (rand_mat >= t)
    return func

def rfcpc_connectivity(c_rate, internal=False): #random fixed coupling prob connectivity
    def func(a_shape, b_shape):
        shape = (*a_shape, *b_shape)
        c_count = int(torch.prod(torch.tensor(shape))*c_rate)
        return rfccc_connectivity(c_count=c_count, internal=internal)(a_shape, b_shape)
    return func

def rfpopp_connectivity(c_rate, internal=False): #random fixed prob of presynaptic partnering connectivity
    def func(a_shape, b_shape):
        c_count = int(torch.prod(torch.tensor(a_shape))*c_rate)
        return rfnopp_connectivity(c_count=c_count, internal=internal)(a_shape, b_shape)
    return func
