"""
Analyzer can make a module analyzable.
"""


from typing import Iterable, Dict, Callable
from typeguard import typechecked
from .monitor import Monitor


@typechecked
class Analyzer:
    """
    Analyzer can make a module analyzable.

    It will add a monitor to the module and will be an interface between the monitor and the module.\
    Also it will provide some decorators to easily manage analysis related subjects.

    Arguments
    ---------
    active : bool
        If be false, analyzer will do nothing.
    """
    def __init__(
        self,
        active: bool = True
    ) -> None:
        if 'analyzable' in self.__dict__:
            raise Exception(f"Object {self} already has an alanyzer.")
        self.analyzable = active
        if self.analyzable:
            self.monitor = Monitor(self)
        
        
    def scout(
        self,
        state_variables: Iterable[str] = [],
        state_calls: Dict[str, Callable] = {},
    ) -> None:
        """
        Adds state variables and state calls to the monitor.

        Arguments
        ---------
        state_variables : Iterable[str]
            The given variables to be added to the monitor state variables.
        state_calls : Dict[str, Callable]
            The given calls to be added to the monitor state calls.

        Returns
        -------
        None
        
        """
        if self.analyzable:
            self.monitor.add_to_state_variables(state_variables)
            self.monitor.add_to_state_calls(state_calls)




def analysis_point(function):
    """
    This decoder specifies the point at which the monitor should start recording at each turn.\
    This happens after executing the decorated function.
    """
    def wrapper(self, *args, **kwargs):
        output = function(self, *args, **kwargs)
        if self.analyzable:
            self.monitor.record_all()
        return output
    return wrapper


def analytics(function):
    """
    This decorator specifies that the decorated function is for analysis and\
    if the analyzer is not active, this function is not applicable.
    """
    def wrapper(self, *args, **kwargs):
        if not self.analyzable:
            raise Exception("The object is not analyzable!")
        return function(self, *args, **kwargs)
    return wrapper