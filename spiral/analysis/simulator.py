"""
"""

from typing import Iterable, Dict, Callable, Any


class DictionaryItemsIterator:
    def __init__(
        self,
        dictionary: Dict[str, Iterable[Any]]:
    ) -> None:
        self.dictionary = dictionary
        self.iter_dictionary = {i: iter(it) for i,it in self.dictionary.items()}
    
    def __next__(
        self
    ) -> Dict[str, Any]:
        output = dict()
        for k in self.iter_dictionary:
            v = next(self.iter_dictionary[k], None)
            if v is None:
                self.iter_dictionary[k] = iter(self.dictionary[k])
                v = next(self.iter_dictionary[k])
            output[k] = v
        return output


class Simulator:
    def __init__(
        self,
        func: Callable
    ) -> None:
        self.func = func


    def simulate(self, inputs={}, times=None):
        inputs = DictionaryItemsIterator(inputs)
        for i in range(times):
            self.func(**next(inputs))