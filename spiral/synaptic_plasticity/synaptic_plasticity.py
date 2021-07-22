"""
"""


from construction_requirements_integrator import CRI, construction_required
from constant_properties_protector import CPP
from typing import Union, Iterable
from typeguard import typechecked
import torch
from spiral.response_function import ResponseFunction
from spiral.learning_rate import SynapticPlasticityRate




@typechecked
class SynapticPlasticity(torch.nn.Module, CRI):
    def __init__(
        self,
        source: Iterable[int] = None,
        target: Iterable[int] = None,
        batch: int = None,
        dt: Union[float, torch.Tensor] = None,
        construction_permission: bool = True,
    ) -> None:
        torch.nn.Module.__init__(self)
        CPP.protect(self, 'source')
        CPP.protect(self, 'target')
        CPP.protect(self, 'batch')
        CPP.protect(self, 'dt')
        CRI.__init__(
            self,
            source=source,
            target=target,
            batch=batch,
            dt=dt,
            ignore_overwrite_error=True,
            construction_permission=construction_permission,
        )


    def __construct__(
        self,
        source: Iterable[int],
        target: Iterable[int],
        batch: int,
        dt: Union[float, torch.Tensor],
    ) -> None:
        self._source = source
        self._target = target
        self._batch = batch
        self.register_buffer("_dt", torch.as_tensor(dt))


    @construction_required
    def __call__(
        self,
        neurotransmitters: torch.Tensor,
        neuromodulators: torch.Tensor,
        action_potential: torch.Tensor,
        synaptic_weights: torch.Tensor,
    ) -> torch.Tensor:
        return torch.as_tensor(0.)


    def reset(
        self
    ) -> None:
        pass




@typechecked
class CompositeSynapticPlasticity(SynapticPlasticity): # don't care about order
    def __init__(
        self,
        synaptic_plasticities: Iterable[SynapticPlasticity],
        source: Iterable[int] = None,
        target: Iterable[int] = None,
        batch: int = None,
        dt: Union[float, torch.Tensor] = None,
        construction_permission: bool = True,
    ) -> None:
        super().__init__(
            source=source,
            target=target,
            batch=batch,
            dt=dt,
            construction_permission=False,
        )
        self.synaptic_plasticities = synaptic_plasticities
        self.set_construction_permission(construction_permission)


    def __construct__(
        self,
        source: Iterable[int],
        target: Iterable[int],
        batch: int,
        dt: Union[float, torch.Tensor],
    ) -> None:
        super().__construct__(
            source=source,
            target=target,
            batch=batch,
            dt=dt,
        )
        for synaptic_plasticity in self.synaptic_plasticities:
            synaptic_plasticity.meet_requirement(source=source)
            synaptic_plasticity.meet_requirement(target=target)
            synaptic_plasticity.meet_requirement(batch=batch)
            synaptic_plasticity.meet_requirement(dt=dt)


    @construction_required
    def __call__(
        self,
        neurotransmitters: torch.Tensor,
        neuromodulators: torch.Tensor,
        action_potential: torch.Tensor,
        synaptic_weights: torch.Tensor,
    ) -> torch.Tensor:
        dw = 0
        for synaptic_plasticity in self.synaptic_plasticities:
            dw = dw + synaptic_plasticity(
                neurotransmitters=neurotransmitters,
                neuromodulators=neuromodulators,
                action_potential=action_potential,
                synaptic_weights=synaptic_weights,
            )
        return dw


    def reset(
        self
    ) -> None:
        for synaptic_plasticity in self.synaptic_plasticities:
            synaptic_plasticity.reset()




@typechecked
class ConvergentSynapticPlasticity(SynapticPlasticity):
    def __init__(
        self,
        tau: Union[float, torch.Tensor] = 250.,
        convergence: Union[float, torch.Tensor] = 0.,
        source: Iterable[int] = None,
        target: Iterable[int] = None,
        batch: int = None,
        dt: Union[float, torch.Tensor] = None,
        construction_permission: bool = True,
    ) -> None:
        super().__init__(
            source=source,
            target=target,
            batch=batch,
            dt=dt,
            construction_permission=construction_permission,
        )
        self.register_buffer("tau", torch.as_tensor(tau))
        self.register_buffer("convergence", torch.as_tensor(convergence))


    @construction_required
    def __call__(
        self,
        neurotransmitters: torch.Tensor,
        neuromodulators: torch.Tensor,
        action_potential: torch.Tensor,
        synaptic_weights: torch.Tensor,
    ) -> torch.Tensor:
        return - (synaptic_weights - self.convergence) * self.dt / self.tau




@typechecked
class STDP(SynapticPlasticity):
    """
    Spike-Time Dependent Plasticity learning rule.
    """
    def __init__(
        self,
        presynaptic_tagging: ResponseFunction,
        postsynaptic_tagging: ResponseFunction,
        ltp_rate: SynapticPlasticityRate = None,
        ltd_rate: SynapticPlasticityRate = None,
        source: Iterable[int] = None,
        target: Iterable[int] = None,
        batch: int = None,
        dt: Union[float, torch.Tensor] = None,
        construction_permission: bool = True,
    ) -> None:
        super().__init__(
            source=source,
            target=target,
            batch=batch,
            dt=dt,
            construction_permission=False,
        )
        self.presynaptic_tagging = presynaptic_tagging
        self.postsynaptic_tagging = postsynaptic_tagging
        self.ltp_rate = SynapticPlasticityRate(rate=0.05) if ltp_rate is None else ltp_rate
        self.ltd_rate = SynapticPlasticityRate(rate=0.01) if ltd_rate is None else ltd_rate
        self.set_construction_permission(construction_permission)


    def __construct__(
        self,
        source: Iterable[int],
        target: Iterable[int],
        batch: int,
        dt: Union[float, torch.Tensor],
    ) -> None:
        super().__construct__(
            source=source,
            target=target,
            batch=batch,
            dt=dt,
        )
        self.presynaptic_tagging.meet_requirement(shape=(self.batch, *self.source))
        self.presynaptic_tagging.meet_requirement(dt=self.dt)
        self.postsynaptic_tagging.meet_requirement(shape=(self.batch, *self.target))
        self.postsynaptic_tagging.meet_requirement(dt=self.dt)
        self.ltp_rate.meet_requirement(dt=self.dt)
        self.ltd_rate.meet_requirement(dt=self.dt)


    @construction_required
    def __call__(
        self,
        neurotransmitters: torch.Tensor,
        neuromodulators: torch.Tensor,
        action_potential: torch.Tensor,
        synaptic_weights: torch.Tensor,
    ) -> torch.Tensor:
        presynaptic_tag = self.presynaptic_tagging(action_potential=neurotransmitters)
        postsynaptic_tag = self.postsynaptic_tagging(action_potential=action_potential)
        ltp_rate = self.ltp_rate(synaptic_weights=synaptic_weights.reshape(1, *self.source, 1, *self.target))
        ltd_rate = self.ltd_rate(synaptic_weights=synaptic_weights.reshape(1, *self.source, 1, *self.target))
        ltp = ltp_rate * presynaptic_tag.reshape(self.batch, *self.source, *[1]*len(self.target)) * action_potential
        ltd = ltd_rate * neurotransmitters.reshape(self.batch *self.source, *[1]*len(self.target)) * postsynaptic_tag
        dw = (ltp - ltd) * self.dt
        return dw.mean([0, len(self.source)])


    def reset(
        self
    ) -> None:
        self.presynaptic_tagging.reset()
        self.postsynaptic_tagging.reset()
        self.ltp_rate.reset()
        self.ltd_rate.reset()