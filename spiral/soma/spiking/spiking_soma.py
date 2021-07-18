"""

"""

from abc import abstractmethod
import torch
from constant_properties_protector import CPP
from spiral.analysis import Analyzer, analysis_point, analytics
from typing import Union, Iterable
from ..soma import Soma




class SpikingSoma(Soma):
    def __init__(
        self,
        name: str,
        shape: Iterable[int] = None,
        dt: Union[float, torch.Tensor] = None,
        analyzable: bool = False,
        construction_permission: bool = True,
    ) -> None:
        super().__init__(
            name=name,
            shape=shape,
            dt=dt,
            construction_permission=construction_permission,
        )
        CPP.protect(self, 'spike')
        Analyzer.__init__(self, analyzable)
        Analyzer.scout(self, state_variables=['spike'])


    def __construct__(
        self,
        shape: Iterable[int],
        dt: Union[float, torch.Tensor],
    ) -> None:
        super().__construct__(
            shape=shape,
            dt=dt,
        )
        self.register_buffer("_spike", torch.zeros(*self.shape, dtype=torch.bool))
                
    
    @abstractmethod
    def _process(
        self,
        inputs: torch.Tensor
    ) -> torch.Tensor:
        pass


    @abstractmethod
    def _fire_axon_hillock(
        self,
        clamps: torch.Tensor = torch.tensor(False),
        unclamps: torch.Tensor = torch.tensor(False),
    ) -> None:
        self._spike = ((self.spike * ~unclamps) + clamps)


    def __propagate_spike(
        self
    ) -> None:
        for axon in self.axons.values():
            axon.forward(self.spike)
        for dendrite in self.dendrites.values():
            dendrite.backward(self.spike)
        

    @analysis_point
    def progress(
        self,
        direct_input: torch.Tensor = torch.tensor(0.),
        clamps: torch.Tensor = torch.tensor(False),
        unclamps: torch.Tensor = torch.tensor(False)
    ) -> None:
        super().progress()
        self._process(inputs=self._integrate_inputs(direct_input=direct_input))
        self._fire_axon_hillock(clamps=clamps, unclamps=unclamps)
        self.__propagate_spike()


    def reset(
        self
    ) -> None:
        self._spike.zero_()
        super().reset()
        if self.analyzable:
            self.monitor.reset()


    @analytics
    def plot_spikes(
        self,
        axes,
        **kwargs
    ) -> None:
        spikes = self.monitor['spike']
        spikes = spikes.reshape(spikes.shape[0], -1)
        time_range = (0, spikes.shape[0])
        x,y = torch.where(spikes)
        x = x*self.dt
        axes.scatter(x, y, **kwargs)
        axes.set_ylabel('spike')
        axes.set_xlabel('time (ms)')
        axes.set_xlim(time_range)
        axes.get_yaxis().set_ticks([])


    @analytics
    def plot_population_activity(
        self,
        axes,
        **kwargs
    ) -> None:
        spikes = self.monitor['spike']
        spikes = spikes.reshape(spikes.shape[0], -1)
        time_range = (0, spikes.shape[0])
        x = torch.arange(*time_range)*self.dt
        y = spikes.sum(axis=1)
        axes.plot(x, y, **kwargs)
        axes.set_xlabel('time (ms)')
        axes.set_ylabel(f'activity (#spikes/{self.dt}ms')
        axes.set_xlim(time_range)