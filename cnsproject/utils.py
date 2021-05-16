import torch

class DII: #Dictionary Items Iterator
    def __init__(self, dictionary):
        self.dictionary = dictionary
        self.iter_dictionary = {i: iter(it) for i,it in self.dictionary.items()}
    
    def __next__(self):
        output = dict()
        for k in self.iter_dictionary:
            v = next(self.iter_dictionary[k], None)
            if v is None:
                self.iter_dictionary[k] = iter(self.dictionary[k])
                v = next(self.iter_dictionary[k])
            output[k] = v
        return output

class SliceMaker(object):
    def __getitem__(self, item):
        return item

def step_function(length, step_index, val0=0, val1=1):
    u = torch.zeros(length) + val0
    u[step_index:] += val1
    return u

def generate_function(length, shape=[], shift=0, noise=0, population_noise=0, slope=0):
    _shift = torch.zeros(length)
    _slope = torch.zeros(length)
    for order,vec in [(shift,_shift),(slope,_slope)]:
        if type(order)==type(.1) or type(order)==type(1):
            order = {0: order}
        if type(order)==type({}):
            for _range,s in order.items():
                start,end = _range,length
                if type(_range)!=type(0):
                    start,end = _range
                vec[start:end] += s
        else:
            vec = order
    output = torch.normal(_slope, noise) if noise>0 else _slope
    output = output.reshape(-1, *[1 for i in shape])
    if population_noise>0:
        output = output + torch.normal(0, population_noise, (length, *shape))
    output = output.cumsum(dim=0)
    output += output+_shift.reshape(-1, *[1 for i in shape])
    return output

def masked_shift(source, mask, shift=1, replace=0):
    if mask.sum()==0:
        return source
    source = source.clone()
    source[:,mask] = torch.cat((source[-shift:,mask], source[:-shift,mask]))
    if replace is not None:
        source[0,mask] = replace
    return source