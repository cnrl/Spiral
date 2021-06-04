import torch
from abc import ABC, abstractmethod
from typing import Union, Iterable
from .axon_sets import AbstractAxonSet

class AbstractNeuromodulatoryTissue(ABC, torch.nn.Module):
    def __init__(
        self,
        name: str,
        dt: float = None,
        config_prohibit: bool = False,
        **kwargs
    ) -> None:
        super().__init__()
        self.configed = False
        self.axons = {}
        self.free_axon_index = 0
        self.register_buffer("d", torch.tensor(0.))
        self.config_prohibit = config_prohibit
        self.name = name
        self.set_dt(dt)


    def config_permit(self):
        return (
            self.dt is not None and
            not self.config_prohibit and
            not self.configed
        )


    def config(self) -> bool:
        if not self.config_permit():
            return False
        self.configed = True
        return True


    def set_dt(self, dt:float):
        if self.configed:
            return False
        self.dt = torch.tensor(dt) if dt is not None else dt
        self.config()
        return True
            

    def add_axon(self, axon_set: AbstractAxonSet) -> None:
        axon_set.set_name(self.name+"_axon_"+str(self.free_axon_index), soft=True)
        axon_set.set_population_shape(())
        axon_set.set_dt(self.dt)
        self.axons[axon_set.name] = axon_set
        self.add_module(axon_set.name, axon_set)
        self.free_axon_index += 1


    def remove_axon(self, name: str) -> None:
        del self.axons[name]

            
    def use(self, other: Union[Iterable, AbstractAxonSet]) -> None:
        if hasattr(other, '__iter__'):
            for o in other:
                self.use(o)
        elif issubclass(type(other), AbstractAxonSet):
            self.add_axon(other)
        else:
            assert False, f"You just can add AbstractAxonSet to neuromodulatory tissues. Your object is {type(other)}"


    def forward(self) -> None:
        for axon_set in self.axons.values():
            axon_set.forward(self.activity())


    def feedback(self) -> None: #compute activity based on inputs
        pass


    def activity(self) -> torch.Tensor:
        return self.d


    def reset(self) -> None:
        self.d.zero_()
        for axon_set in self.axons.values():
            axon_set.reset()


    def __str__(self):
        return f"[{self.name}] {', '.join([a.__str__() for a in self.axons.values()])}"




class AbstractDopaminergicTissue(AbstractNeuromodulatoryTissue):
    def __init__(
        self,
        name: str,
        **kwargs
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.register_buffer("da", torch.tensor(0.))


    def forward(self) -> None:
        self.da.zero_()
        super().forward()

    
    def feedback(self, da: Union[float, torch.Tensor]) -> None:
        self.da = torch.tensor(da)


    def reset(self) -> None:
        self.da.zero_()
        super().reset()




class SimpleDopaminergicTissue(AbstractDopaminergicTissue):
    def __init__(
        self,
        name: str,
        tau: Union[float, torch.Tensor] = 7.,
        **kwargs
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.register_buffer("tau", torch.tensor(tau))


    def forward(self) -> None:
        self.d += (self.da - self.d / self.tau) * self.dt
        super().forward()




class ImpulseDopaminergicTissue(AbstractDopaminergicTissue):
    def __init__(
        self,
        name: str,
        **kwargs
    ) -> None:
        super().__init__(name=name, **kwargs)


    def forward(self) -> None:
        self.d = self.da.clone()
        super().forward()




class FlatDopaminergicTissue(AbstractDopaminergicTissue):
    def __init__(
        self,
        name: str,
        time_window: Union[float, torch.Tensor] = 7.,
        config_prohibit: bool = False,
        **kwargs
    ) -> None:
        super().__init__(name=name, config_prohibit=True, **kwargs)
        self.time_window = time_window
        self.config_prohibit = config_prohibit
        self.config()


    def config(self) -> bool:
        if not self.config_permit():
            return False
        self.length = int(self.time_window//self.dt)
        self.register_buffer("da_history", torch.zeros((self.length), dtype=torch.bool))
        self.configed = True
        return True


    def forward(self) -> None:
        self.da_history = torch.cat([self.da.unsqueeze(0), self.da_history[:-1]])
        self.d = self.da_history.sum(axis=0)
        super().forward()

    
    def reset(self) -> None:
        self.da_history.zero_()
        super().reset()