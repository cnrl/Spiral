"""
Monitors can record things over time.
"""


import torch
from typing import Iterable, Dict, Callable, Any
from typeguard import typechecked




@typechecked
class Monitor:
    """
    An class to record variables or output of function calls of an object over time.

    Properties
    ----------
    obj : Object
        The target object to record things of it.
    device : str
        Device that the monitor is on.
    recording : Dict[str, Iterable[torch.Tensor]]
        Container of recorded things.
    state_variables : Iterable[str]
        Object properties that asked to be recorded
    state_calls : Dict[str, Callable]
        Functions that asked to be called and record their outputs.

    Arguments
    ---------
    obj : Object, Necessary
        The target object to record things of it.
    device : str, Optional, default: "cpu"
        Allow the monitor to be on different device separate from Network device.
    """
    def __init__(
        self,
        obj,
        device: str = "cpu",
    ) -> None:
        self.obj = obj
        self.device = device
        self.recording = {}
        self.state_variables = []
        self.state_calls = {}
        self.reset()

    
    def add_to_state_variables(
        self,
        state_variables: Iterable[str]
    ) -> None:
        """
        Adds given variables to the state variables (to be recorded).
        
        Arguments
        ---------
        state_variables : Iterable[str]
            The given variables to be added to the state variables.
        
        Returns
        -------
        None
        
        """
        self.state_variables += list(state_variables)
        self.reset()


    def add_to_state_calls(
        self,
        state_calls: Dict[str, Callable]
    ) -> None:
        """
        Adds given calls to the state calls (to be recorded).
        
        Arguments
        ---------
        state_calls : Dict[str, Callable]
            The given calls to be added to the state calls.
        
        Returns
        -------
        None
        
        """
        self.state_calls.update(dict(state_calls))
        self.reset()


    def get(
        self,
        variable: str
    ) -> torch.Tensor:
        """
        Returns the recording values for the given name.
        
        Arguments
        ---------
        variable : str
            Name of asked recordings.
        
        Returns
        -------
        recorded : torch.Tensor
            The recording values for the given name.
        
        """
        return torch.as_tensor([a.tolist() for a in self.recording[variable]])


    def __getitem__(
        self,
        variable: str
    ) -> torch.Tensor:
        """
        Returns the recording values for the given name.
        
        Arguments
        ---------
        variable : str
            Name of asked recordings.
        
        Returns
        -------
        recorded : torch.Tensor
            The recording values for the given name.
        
        """
        return self.get(variable)


    def record(
        self,
        name: str,
        data: torch.Tensor,
    ) -> None:
        """
        Push data to the recorded values for the `name`.
        
        Arguments
        ---------
        name : str
            Name of the recordings value.
        data : torch.Tensor
            The recordings value.
        
        Returns
        -------
        None
        
        """
        self.recording[name].append(data.detach().clone().to(self.device))


    def record_all(
        self
    ) -> None:
        """
        Records all the state variables and output of all state calls.
        
        Returns
        -------
        None
        
        """
        for name in self.state_variables:
            data = getattr(self.obj, name)
            self.record(name, data)
        for name,call in self.state_calls.items():
            data = call()
            self.record(name, data)


    def reset(
        self
    ) -> None:
        """
        Refractor and reset the axon and related moduls.
        
        Returns
        -------
        None
        
        """
        self.recording = {}
        for key in self.state_variables+list(self.state_calls.keys()):
            self.recording[key] = []