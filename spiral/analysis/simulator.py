"""
Provides simulator to run things over time.
"""


from typing import Iterable, Dict, Callable, Union, Any
from .analyzer import TimeAnalysis
from typeguard import typechecked




@typechecked
@TimeAnalysis()
class DictionaryItemsIterator:
    """
    An auxiliary data structure for iterating a dictionary of iteratables.

    Properties
    ----------
    dictionary : Dict[str, Iterable or Dict of Iterables]
        Prime dictionary.

    Arguments
    ---------
    dictionary : Dict[str, Iterable or Dict of Iterables]
        Prime dictionary.\
        The value should be iterables. In each call, one iteration of each value will be returned.\
        It is possible to set a value to a dictionary of iterables and make a recursive iteration over dictionaries.
    """
    def __init__(
        self,
        dictionary: Dict[str, Union[Iterable[Any], dict]],
    ) -> None:
        self.dictionary = dictionary
        self.__iter_dictionary = {
            i: iter(it) if type(it) is not dict else DictionaryItemsIterator(it)
            for i,it in self.dictionary.items()
        }


    def __next__(
        self
    ) -> Dict[str, Any]:
        """
        Iterates one step over the given dictionary.

        Returns
        -------
        output: Dict
            Output of iteration.
        
        """
        output = dict()
        for k in self.__iter_dictionary:
            v = next(self.__iter_dictionary[k], None)
            if v is None:
                self.__iter_dictionary[k] = iter(self.dictionary[k])
                v = next(self.__iter_dictionary[k])
            output[k] = v
        return output




@typechecked
@TimeAnalysis()
class Simulator:
    """
    An class to call a function over time.

    Properties
    ----------
    func : Callable
        The function to be called over time.

    Arguments
    ---------
    func : Callable
        The function to be called over time.
    """
    def __init__(
        self,
        func: Callable
    ) -> None:
        self.func = func


    def simulate(
        self,
        times: int,
        inputs={}
    ) -> None:
        """
        Calls the given function for `times` times.\
        It will feed the function using iteration over the given inputs.\
        Read more about iteration over a dictionary of iterables in Spiral.DictionaryItemsIterator module documentations.

        Arguments
        ---------
        times : int
            Number of function calls.
        inputs :  Dict[str, Iterable or Dict of Iterables], Optional, default: {}
            A dictionary of iterables as function inputs.\
            The keys should be the name of function arguments.\
            The value should be iterables. In each call, one iteration of each value will be passed to the function.\
            It is possible to set a value to a dictionary of iterables and make a recursive iteration over dictionaries.\
            By doing this, a dictionary will be passed to the function as corresponding argument.

        Returns
        -------
        None
        
        """
        inputs = DictionaryItemsIterator(inputs)
        for i in range(times):
            self.func(**next(inputs))