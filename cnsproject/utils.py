import numpy as np

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

def step_function(length, step_index, val0=0, val1=1):
    u = np.zeros(length) + val0
    u[step_index:] += val1
    return u