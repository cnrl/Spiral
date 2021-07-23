"""
"""


import torch
from typing import Union, Iterable, Callable
from typeguard import typechecked
from abc import ABC, abstractmethod
from constant_properties_protector import CPP
from construction_requirements_integrator import CRI, construction_required
from spiral.analysis import Analyzer, analysis_point, analytics
from spiral.synaptic_plasticity.synaptic_plasticity import SynapticPlasticity




@typechecked
class Dendrite(torch.nn.Module, CRI, ABC):
    def __init__(
        self,
        name: str = None,
        shape: Iterable[int] = None,
        batch: int = None,
        spine: Iterable[int] = None,
        dt: float = None,
        plasticity_model: SynapticPlasticity = None,
        plasticity: bool = True,
        analyzable: bool = False,
        construction_permission: bool = True,
    ) -> None:
        torch.nn.Module.__init__(self)
        CPP.protect(self, 'name')
        CPP.protect(self, 'shape')
        CPP.protect(self, 'spine')
        CPP.protect(self, 'dt')
        CPP.protect(self, 'batch')
        CPP.protect(self, 'neurotransmitters')
        CPP.protect(self, 'neuromodulators')
        CPP.protect(self, 'action_potential')
        self.plasticity_model = SynapticPlasticity() if plasticity_model is None else plasticity_model
        self.plasticity = plasticity
        Analyzer.__init__(self, analyzable)
        Analyzer.scout(self, state_calls={'transmit_current': self.transmit_current})
        CRI.__init__(
            self,
            name=name,
            shape=shape,
            batch=batch,
            spine=spine,
            dt=dt,
            construction_permission=construction_permission,
            ignore_overwrite_error=True,
        )


    @property
    @abstractmethod
    def synaptic_weights(
        self
    ) -> torch.Tensor:
        pass


    def __construct__(
        self,
        name: str,
        shape: Iterable[int],
        batch: int,
        spine: Iterable[int],
        dt: Union[float, torch.Tensor],
    ) -> None:
        self._name = name
        self._shape = (*shape,)
        self._spine = (*spine,)
        self._batch = batch
        self.register_buffer("_dt", torch.as_tensor(dt))
        self.plasticity_model.meet_requirement(source=spine)
        self.plasticity_model.meet_requirement(target=shape)
        self.plasticity_model.meet_requirement(batch=batch)
        self.plasticity_model.meet_requirement(dt=self.dt)


    @analysis_point
    def forward(
        self,
        neurotransmitters: torch.Tensor,
        neuromodulators: torch.Tensor,
    ) -> None:
        self._neurotransmitters = neurotransmitters
        self._neuromodulators = neuromodulators


    def _update_synaptic_weights(
        self,
        synaptic_weights_plasticity: torch.Tensor,
    ) -> None:
        pass


    @construction_required
    def backward(
        self,
        action_potential: torch.Tensor
    ) -> None:
        self._action_potential = action_potential
        synaptic_weights_plasticity = self.plasticity_model(
            neurotransmitters=self.neurotransmitters,
            neuromodulators=self.neuromodulators,
            action_potential=self.action_potential,
            synaptic_weights=self.synaptic_weights,
        )
        if self.plasticity:
            self._update_synaptic_weights(synaptic_weights_plasticity)


    @abstractmethod
    def transmit_current(
        self,
    ) -> torch.Tensor:
        pass


    def reset(
        self
    ) -> None:
        if hasattr(self, '_neurotransmitters'):
            del self._neurotransmitters
        if hasattr(self, '_neuromodulators'):
            del self._neuromodulators
        if hasattr(self, '_action_potential'):
            del self._action_potential
        self.plasticity_model.reset()
        if self.analyzable:
            self.monitor.reset()


    @analytics
    def plot_transmiting_current(
        self,
        axes,
        **kwargs
    ) -> None:
        """
        Draw a plot of transmiting current on `axes`.

        Arguments
        ---------
        axes : Matplotlib plotable module
            The axes to draw on.
        **kwargs : keyword arguments
            kwargs will be directly passed to matplotlib plot function.
        
        Returns
        -------
        None
        
        """
        y = self.monitor['transmit_current'].reshape(self.monitor['transmit_current'].shape[0],-1)
        time_range = (0, y.shape[0])
        x = torch.arange(*time_range)*self.dt
        population_alpha = 1/y.shape[1]
        aggregated = y.mean(axis=1)
        axes.plot(x, aggregated, color='blue', **kwargs)
        axes.plot(x, y, alpha=population_alpha, color='blue')
        axes.set_ylabel('transmiting current')
        axes.set_xlabel('time (ms)')
        axes.set_xlim(time_range)




@typechecked
class LinearDendrite(Dendrite):
    def __init__(
        self,
        name: str = None,
        shape: Iterable[int] = None,
        batch: int = None,
        spine: Iterable[int] = None,
        initial_weights: Union[float, torch.Tensor, Callable] = torch.rand,
        maximum_weight: Union[float, torch.Tensor] = 1.,
        minimum_weight: Union[float, torch.Tensor] = 0.,
        dt: float = None,
        plasticity_model: SynapticPlasticity = None,
        plasticity: bool = True,
        analyzable: bool = False,
        construction_permission: bool = True,
    ) -> None:
        super().__init__(
            name=name,
            shape=shape,
            batch=batch,
            spine=spine,
            dt=dt,
            plasticity_model=plasticity_model,
            plasticity=plasticity,
            analyzable=analyzable,
            construction_permission=False,
        )
        self.register_buffer('max', torch.as_tensor(maximum_weight))
        self.register_buffer('min', torch.as_tensor(minimum_weight))
        self.add_to_construction_requirements(initial_weights=initial_weights)
        CPP.protect(self, 'w')
        self.set_construction_permission(construction_permission)
        Analyzer.scout(self, state_variables=['w'])


    @property
    def synaptic_weights(
        self
    ) -> torch.Tensor:
        return self.w


    def __construct__(
        self,
        name: str,
        shape: Iterable[int],
        batch: int,
        spine: Iterable[int],
        initial_weights: Union[float, torch.Tensor, Callable],
        dt: Union[float, torch.Tensor],
    ) -> None:
        super().__construct__(
            name=name,
            shape=shape,
            batch=batch,
            spine=spine,
            dt=dt,
        )
        if callable(initial_weights):
            initial_weights = initial_weights((*self.spine, *self.shape))
        initial_weights = torch.as_tensor(initial_weights)
        if initial_weights.numel()==1:
            initial_weights = initial_weights.reshape(*self.spine, *self.shape)
        if initial_weights.shape!=(*self.spine, *self.shape):
            raise Exception(f"`initial_weights` must be in shape {*self.spine, *self.shape}")
        self.register_buffer("_w", initial_weights)
        self._keep_weight_limits()


    def _keep_weight_limits(self):
        self._w[self.w>self.max] = self.max
        self._w[self.w<self.min] = self.min


    @construction_required
    def transmit_current(
        self,
    ) -> torch.Tensor:
        output = self.neurotransmitters.reshape(self.batch, *self.spine, *[1]*len(self.shape))
        output = output * self.synaptic_weights
        output = output.sum(axis=list(range(len(self.spine)+1)))
        return output


    def _update_synaptic_weights(
        self,
        synaptic_weights_plasticity: torch.Tensor,
    ) -> None:
        self._w += synaptic_weights_plasticity
        self._keep_weight_limits()


    @analytics
    def plot_synaptic_weights(
        self,
        axes,
        **kwargs
    ) -> None:
        y = self.monitor['w'].reshape(self.monitor['w'].shape[0],-1)
        time_range = (0, y.shape[0])
        x = torch.arange(*time_range)*self.dt
        population_alpha = 1/y.shape[1]
        axes.plot(x, y, alpha=population_alpha)
        axes.set_ylabel('synaptic weights')
        axes.set_xlabel('time (ms)')
        axes.set_xlim(time_range)