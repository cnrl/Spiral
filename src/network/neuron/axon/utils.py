import torch

def masked_shift(source, mask, shift=1, replace=0):
    if mask.sum()==0:
        return source
    source = source.clone()
    source[:,mask] = torch.cat((source[-shift:,mask], source[:-shift,mask]))
    if replace is not None:
        source[0,mask] = replace
    return source