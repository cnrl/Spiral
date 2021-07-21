"""
Module for connections between neural populations.
"""


from typing import Union, Iterable
from typeguard import typechecked
import torch
from construction_requirements_integrator import CRI, construction_required
from spiral.analysis import Analyzer, analysis_point, analytics
from spiral.synaptic_plasticity import SynapticPlasticity
# from .weight_initialization import constant_initialization
# from ..filter.filter import AbstractFilter




@typechecked
class Dendrite(torch.nn.Module, CRI):
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
        CPP.protect(self, 'neurotransmitters')
        CPP.protect(self, 'neuromodulators')
        CPP.protect(self, 'action_potential')
        self.plasticity_model = SynapticPlasticity() if plasticity_model is None else plasticity_model
        self.plasticity = plasticity
        Analyzer.__init__(self, analyzable)
        Analyzer.scout(self, state_calls=['transmit_current'])
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
    def synaptic_weights(
        self
    ) -> torch.Tensor:
        return torch.as_tensor(1.)


    def __construct__(
        self,
        name: str,
        shape: Iterable[int],
        batch: int,
        spine: Iterable[int],
        dt: Union[float, torch.Tensor],
    ) -> None:
        self._name = name
        self._shape = (batch, *shape)
        self._spine = (batch, *spine)
        self.register_buffer("_dt", torch.as_tensor(dt))
        self.plasticity_model.meet_requirement(source=spine)
        self.plasticity_model.meet_requirement(target=shape)
        self.plasticity_model.meet_requirement(batch=batch)
        self.plasticity_model.meet_requirement(dt=self.dt)


    @analysis_point
    def forward(
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
            self.neurotransmitters,
            self.neuromodulators,
            self.action_potential,
            self.synaptic_weights,
        )
        if self.plasticity:
            self._update_synaptic_weights(synaptic_weights_plasticity)


    @construction_required
    def transmit_current(
        self,
    ) -> torch.Tensor:
        return (
            self.synaptic_weights * (
                self.neurotransmitters
            ).reshape(*self.spine, *[1]*len(self.shape))
        ).sum(axis=list(range(len(self.spine))))


    def reset(
        self
    ) -> torch.Tensor:
        if hasattr(self, '_neurotransmitters'):
            del self._neurotransmitters
        if hasattr(self, '_neuromodulators'):
            del self._neuromodulators
        if hasattr(self, '_action_potential'):
            del self._action_potential
        self.synaptic_plasticity.reset()




@typechecked
class LinearDendrite(Dendrite):
    def __init__(
        self,
        name: str = None,
        shape: Iterable[int] = None,
        batch: int = None,
        spine: Iterable[int] = None,
        initial_weights: torch.Tensor = torch.as_tensor([]),
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
            construction_required=False,
        )
        self.register_buffer('max', torch.as_tensor(maximum_weight))
        self.register_buffer('min', torch.as_tensor(minimum_weight))
        self.add_to_construction_requirements(initial_weights=initial_weights)
        CPP.protect(self, 'w')
        self.set_construction_permission(construction_permission)
        Analyzer.scout(self, state_calls=['w'])


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
        initial_weights: torch.Tensor,
        dt: Union[float, torch.Tensor],
    ) -> None:
        self.__construct__(
            name=name,
            shape=shape,
            batch=batch,
            spine=spine,
            dt=dt,
        )
        if initial_weights.numel()==0:
            initial_weights = torch.rand(*self.spine[1:], *self.shape[1:])*(self.max-self.min) + self.min
        if initial_weights.shape!=(*self.spine[1:], *self.shape[1:]):
            raise Exception(f"`initial_weights` must be in shape {*self.spine[1:], *self.shape[1:]}")
        self.register_buffer("_w", initial_weights)


    def _update_synaptic_weights(
        self,
        synaptic_weights_plasticity: torch.Tensor,
    ) -> None:
        self._w += synaptic_weights_plasticity