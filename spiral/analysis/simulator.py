"""
"""


from typing import Iterable, Dict, Callable, Union, Any
from typeguard import typechecked




@typechecked
class DictionaryItemsIterator:
    def __init__(
        self,
        dictionary: Dict[str, Union[Iterable[Any], dict]],
    ) -> None:
        self.dictionary = dictionary
        self.iter_dictionary = {
            i: iter(it) if type(it) is not dict else DictionaryItemsIterator(it)
            for i,it in self.dictionary.items()
        }


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




@typechecked
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