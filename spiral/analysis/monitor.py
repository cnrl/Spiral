"""
"""

from typing import Iterable, Dict, Callable, Any
import torch


class Monitor:
    def __init__(
        self,
        obj,
        state_variables: Iterable[str] = [],
        state_calls: Dict[str, Callable] = {},
        device: str = "cpu",
    ) -> None:
        self.obj = obj
        self.device = device
        self.recording = {}
        self.state_variables = []
        self.state_calls = {}
        self.add_to_state_variables(state_variables)
        self.add_to_state_calls(state_calls)
        self.reset()

    
    def add_to_state_variables(
        self,
        state_variables: Iterable[str]
    ) -> None:
        self.state_variables += list(state_variables)
        self.reset()


    def add_to_state_calls(
        self,
        state_calls: Dict[str, Callable]
    ) -> None:
        self.state_calls.update(dict(state_calls))
        self.reset()


    def get(
        self,
        variable: str
    ) -> torch.Tensor:
        return torch.tensor([a.tolist() for a in self.recording[variable]])


    def __getitem__(
        self,
        variable: str
    ) -> torch.Tensor:
        return self.get(variable)


    def record(
        self,
        name: str,
        data: torch.Tensor,
    ) -> None:
        self.recording[name].append(
            torch.empty_like(
                data,
                device=self.device,
                requires_grad=False
            ).copy_(
                data,
                non_blocking=True
            )
        )


    def record_all(
        self
    ) -> None:
        for name in self.state_variables:
            data = getattr(self.obj, name)
            self.record(name, data)
        for name,call in self.state_calls.items():
            data = call()
            self.record(name, data)


    def reset(
        self
    ) -> None:
        self.recording = {}
        for key in self.state_variables+list(self.state_calls.keys()):
            self.recording[key] = []