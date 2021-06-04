import torch
from functools import wraps

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

class Container:
    def __init__(self, dictionary):
        self.__dict__ = dictionary

class Serializer:
    def __init__(self, objs):
        self.objs = objs
        for name in [method_name
                     for method_name
                     in dir(objs[0])
                     if callable(getattr(objs[0], method_name))
                     and method_name[0]!='_']:
            setattr(self, name, self.serialize_function(name))
            
    def serialize_function(self, func):
        @wraps(func)
        def wrapper(*args, **kw):
            return [getattr(obj, func)(*args, **kw) for obj in self.objs]
        return wrapper
            
    def __add__(self, other):
        if type(other) is Serializer:
            return Serializer(self.objs+other.objs)
        else:
            return Serializer(self.objs+[other])
        
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


## this will be removed soon ###########
def convolution2d(image, kernel, bias=0):
    kernel = kernel.reshape(kernel.shape[-2], kernel.shape[-1])
    m, n = kernel.shape
    assert m==n, "kernel shape must be squared"
    image = image.reshape(image.shape[-2], image.shape[-1])
    y, x = image.shape
    y = y - m + 1
    x = x - m + 1
    new_image = torch.zeros((y,x))
    for i in range(y):
        for j in range(x):
            new_image[i][j] = torch.sum(image[i:i+m, j:j+m]*kernel) + bias
    new_image = new_image.view(1,1,*new_image.shape)
    return new_image

def conv2d(kernel, bias=0, **args):
    return lambda image: convolution2d(image, kernel, args.get('bias',0))
########################################