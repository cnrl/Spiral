"""

"""

from abc import abstractmethod
import torch
from spiral.analysis import Analyzer, analysis_point, analytics
from ..soma import Soma




class SpikingSoma(Soma):
    def __init__(
        self,
        name: str,
        analyzable: bool = False,
        **kwargs
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.protect_properties(['spike'])
        Analyzer.__init__(self, analyzable)
        Analyzer.scout(self, state_variables=['spike'])


    def __construct__(
        self,
        **kwargs,
    ) -> None:
        super().__construct__(**kwargs)
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
        unclamps: torch.Tensor = torch.tensor(False)
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
        self._process(self.__integrate_inputs(direct_input=direct_input))
        self._fire_axon_hillock(clamps=clamps, unclamps=unclamps)
        self.__propagate_spike()


    def reset(
        self
    ) -> None:
        self._spike.zero_()
        super().reset()


    @analytics
    def plot_spikes(
        self,
        axes,
        **kwargs
    ) -> None:
        spikes = self.monitor['spike']
        spikes = spikes.reshape(spikes.shape[0], -1)
        time_range = (0, spikes.shape[0])
        x,y = np.where(spikes)
        x *= self.dt
        axes.scatter(x, y, **kwargs)
        axes.set_xlabel('time (ms)')
        axes.set_xlim(time_range)
        axes.get_yaxis().set_ticks([])


    @analytics
    def population_activity(
        self,
        axes,
        **kwargs
    ) -> None:
        spikes = self.monitor['spike']
        spikes = spikes.reshape(spikes.shape[0], -1)
        time_range = (0, spikes.shape[0])
        x = np.arange(time_range)*self.dt
        y = spikes.sum(axis=1)
        axes.plot(x, y, **kwargs)
        axes.set_xlabel('time (ms)')
        axes.set_ylabel(f'activity (#spikes/{self.dt}ms')
        axes.set_xlim(time_range)