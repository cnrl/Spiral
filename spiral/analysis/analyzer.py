from . import Monitor
from typing import Iterable, Dict, Callable

class Analyzer:
    def __init__(
        self,
        active: bool = True
    ) -> None:
        self.analyzable = active
        self.monitor = Monitor(self)
        
        
    def scout(
        self,
        state_variables: Iterable[str] = [],
        state_calls: Dict[str, Callable] = {},
    ) -> None:
        if self.analyzable:
            self.monitor.add_to_state_variables(state_variables)
            self.monitor.add_to_state_calls(state_calls)




def analysis_point(function):
    def wrapper(self, *args, **kwargs):
        output = function(self, *args, **kwargs)
        if self.analyzable:
            self.monitor.record_all()
        return output
    return wrapper


def analytics(function):
    def wrapper(self, *args, **kwargs):
        if not self.analyzable:
            raise Exception("The object is not analyzable!")
        return function(self, *args, **kwargs)
    return wrapper