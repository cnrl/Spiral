from abc import ABC, abstractmethod
from typing import Union

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


    @abstractmethod
    def forward(self) -> None:
        pass


    def feedback(self) -> None:
        pass


    @abstractmethod
    def neuromodulators(self) -> torch.Tensor:
        pass


    def reset(self) -> None:
        pass



class SimpleDopaminergicTissue(AbstractNeuromodulatoryTissue):
    def __init__(
        self,
        name: str,
        tau: Union[float, torch.Tensor] = 7.,
        **kwargs
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.tau = torch.tensor(tau)
        self.d = torch.tensor(0)
        self.da = torch.tensor(0)


    def forward(self) -> None:
        self.d += (self.da - self.d / self.tau) * self.dt
        self.da = torch.tensor(0)
        super.forward()

    
    def feedback(self, da: Union[float, torch.Tensor]) -> None:
        self.da = torch.tensor(da)


    def neuromodulators(self) -> torch.Tensor:
        return self.d


    def reset(self) -> None:
        self.d = torch.tensor(0)
        self.da = torch.tensor(0)
        super().reset()




class FlatDopaminergicTissue(AbstractNeuromodulatoryTissue):
    def __init__(
        self,
        name: str,
        time: Union[float, torch.Tensor] = 7.,
        config_prohibit: bool = False,
        **kwargs
    ) -> None:
        super().__init__(name=name, config_prohibit=True, **kwargs)
        self.time = time
        self.config_prohibit = config_prohibit
        self.config()


    def config(self) -> bool:
        if not self.config_permit():
            return False
        self.length = int(self.time//self.dt)
        self.register_buffer("spike_history", torch.zeros((self.length), dtype=torch.bool))
        self.configed = True
        return True


    def forward(self) -> None:
        self.spike_history = torch.cat([self.da.unsqueeze(0), self.spike_history[:-1]])
        super.forward()

    
    def feedback(self, da: Union[float, torch.Tensor]) -> None:
        self.da = torch.tensor(da)


    def neuromodulators(self) -> torch.Tensor:
        return self.spike_history.sum(axis=0)


    def reset(self) -> None:
        self.spike_history.zero_()
        super().reset()