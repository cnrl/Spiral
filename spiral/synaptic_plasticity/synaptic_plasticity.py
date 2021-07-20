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
        dt: Union[float, torch.Tensor] = None,
        construction_permission: bool = True,
    ) -> None:
        torch.nn.Module.__init__(self)
        CPP.protect(self, 'source')
        CPP.protect(self, 'target')
        CPP.protect(self, 'dt')
        CRI.__init__(
            self,
            source=shape,
            target=target,
            dt=dt,
            ignore_overwrite_error=True,
            construction_permission=construction_permission,
        )


    def __construct__(
        self,
        source: Iterable[int],
        target: Iterable[int],
        dt: Union[float, torch.Tensor],
    ) -> None:
        self._source = source
        self._target = target
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
        dt: Union[float, torch.Tensor] = None,
        construction_permission: bool = True,
    ) -> None:
        super().__init__(
            source=source,
            target=target,
            dt=dt,
            construction_permission=False,
        )
        self.synaptic_plasticities = synaptic_plasticities
        self.set_construction_permission(construction_permission)


    def __construct__(
        self,
        source: Iterable[int],
        target: Iterable[int],
        dt: Union[float, torch.Tensor],
    ) -> None:
        super().__construct__(
            source=source,
            target=target,
            dt=dt,
        )
        for synaptic_plasticity in self.synaptic_plasticities:
            synaptic_plasticity.meet_requirement(source=source)
            synaptic_plasticity.meet_requirement(target=target)
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
        dt: Union[float, torch.Tensor] = None,
        construction_permission: bool = True,
    ) -> None:
        super().__init__(
            source=source,
            target=target,
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
        dt: Union[float, torch.Tensor] = None,
        construction_permission: bool = True,
    ) -> None:
        super().__init__(
            source=source,
            target=target,
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
        dt: Union[float, torch.Tensor],
    ) -> None:
        super().__construct__(
            source=source,
            target=target,
            dt=dt,
        )
        self.presynaptic_tagging.meet_requirement(shape=source)
        self.presynaptic_tagging.meet_requirement(dt=dt)
        self.postsynaptic_tagging.meet_requirement(shape=target)
        self.postsynaptic_tagging.meet_requirement(dt=dt)
        self.ltp_rate.meet_requirement(dt=dt)
        self.ltd_rate.meet_requirement(dt=dt)


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
        ltp_rate = self.ltp_rate(synaptic_weights=synaptic_weights)
        ltd_rate = self.ltd_rate(synaptic_weights=synaptic_weights)
        ltp = ltp_rate * presynaptic_tag.reshape(*self.source, *[1]*len(self.target)) * action_potential
        ltd = ltd_rate * neurotransmitters.reshape(*self.source, *[1]*len(self.target)) * postsynaptic_tag
        return (ltp - ltd) * self.dt


    def reset(
        self
    ) -> None:
        self.presynaptic_tagging.reset()
        self.postsynaptic_tagging.reset()
        self.ltp_rate.reset()
        self.ltd_rate.reset()




# class KernelSTDP(AbstractKernelWeightLRE):
#     def __init__(
#         self,
#         name: str = None,
#         dims: int = 2,
#         pre_traces: AbstractSynapticTagger = None,
#         post_traces: AbstractSynapticTagger = None,
#         ltp_wdlr: Callable = stdp_wdlr(.1), #LTP weight dependent learning rate
#         ltd_wdlr: Callable = stdp_wdlr(.1), #LTD weight dependent learning rate
#         config_prohibit: bool = False,
#         **kwargs
#     ) -> None:  
#         super().__init__(config_prohibit=True, name=name, **kwargs)
#         self.dims = dims
#         self.pre_traces = pre_traces if pre_traces is not None else STDPST()
#         self.post_traces = post_traces if post_traces is not None else STDPST()
#         self.ltp_wdlr = ltp_wdlr
#         self.ltd_wdlr = ltd_wdlr
#         self.config_prohibit = config_prohibit
#         self.config()

    
#     def config(self) -> bool:
#         if not super().config():
#             return False
#         self.pre_traces.set_shape(self.synapse.axon.population_shape)
#         self.post_traces.set_shape(self.synapse.dendrite.population_shape)
#         self.pre_traces.set_dt(self.dt)
#         self.post_traces.set_dt(self.dt)
#         return True


#     def compute_lrs(self) -> tuple: # ltp_lr,ltd_lr
#         w = self.synapse.dendrite.filter.core.weight.data
#         wmin = self.synapse.dendrite.wmin
#         wmax = self.synapse.dendrite.wmax
#         ltp_lr = self.ltp_wdlr(w, wmin, wmax)
#         ltd_lr = self.ltd_wdlr(w, wmin, wmax)
#         return ltp_lr,ltd_lr


#     def presynaptic_process(self, tensor): ##### it need a better name!!!!
#         if not self.synapse.dendrite.filter.channel_inputing:
#             tensor = tensor.unsqueeze(0)
#         kernel_size = self.synapse.dendrite.filter.core.weight.data.shape[-len(tensor.shape)+1:]
#         stride = self.synapse.dendrite.filter.core.stride
#         padding = self.synapse.dendrite.filter.core.padding
#         for i,pad in enumerate(padding):
#             shape = list(tensor.shape)
#             shape[i+1] = pad
#             tensor = torch.cat([torch.zeros(shape),tensor,torch.zeros(shape)], axis=i+1)
#         for i,strd in enumerate(stride):
#             tensor = tensor.unfold(i+1,kernel_size[i],strd)
#         tensor = tensor.unsqueeze(0)
#         return tensor


#     def postsynaptic_process(self, tensor): ##### it need a better name!!!!
#         if not self.synapse.dendrite.filter.channel_outputing:
#             tensor = tensor.unsqueeze(0)
#         tensor = tensor.reshape(tensor.shape[0], 1, *tensor.shape[1:])
#         tensor = tensor.reshape(*tensor.shape, *[1]*(len(tensor.shape)-2))
#         return tensor

    
#     def compute_updatings(self) -> torch.Tensor:
#         ltp_lr,ltd_lr = self.compute_lrs()
        
#         ltp = self.presynaptic_process(self.pre_traces.traces()) * self.postsynaptic_process(self.synapse.dendrite.spikes())
#         ltp = ltp_lr * ltp.sum(axis=list(range(2,len(ltp_lr.shape))))

#         ltd = self.postsynaptic_process(self.post_traces.traces()) * self.presynaptic_process(self.synapse.axon.spikes())
#         ltd = ltd_lr * ltd.sum(axis=list(range(2,len(ltd_lr.shape))))
        
#         dw = self.dt * (ltp - ltd)
#         return dw


#     def forward(self, direct_input: torch.Tensor = torch.as_tensor(0.)) -> None:
#         self.pre_traces.forward(self.synapse.axon.spikes())
#         self.post_traces.forward(self.synapse.dendrite.spikes())
#         super().forward()


#     def reset(self) -> None:
#         self.pre_traces.reset()
#         self.post_traces.reset()
#         super().reset()




# class AbstractNeuromodulatoryLRE(CombinableLRE):
#     def __init__(
#         self,
#         name: str = None,
#         **kwargs
#     ) -> None:
#         super().__init__(name=name, **kwargs)
#         self.axons = {}
#         self.register_buffer("neuromodulators", torch.as_tensor(0.))
            

#     def add_axon(self, axon_set: Union[AbstractAxonSet, Iterable]) -> None:
#         if hasattr(axon_set, '__iter__'):
#             for o in axon_set:
#                 self.usadd_axone(o)
#         else:
#             self.axons[axon_set.name] = axon_set


#     def remove_axon(self, name: str) -> None:
#         del self.axons[name]


#     def collect_neuromodulators(self, direct_input: torch.Tensor = torch.as_tensor(0.)):
#         neuromodulators = direct_input
#         for axon_set in self.axons.values():
#             neuromodulators = neuromodulators + axon_set.neurotransmitters()
#         return neuromodulators


#     def forward(self, direct_input: torch.Tensor = torch.as_tensor(0.)) -> None:
#         self.neuromodulators = self.collect_neuromodulators(direct_input=direct_input)


#     def reset(self) -> None:
#         self.neuromodulators.zero_()
#         super().reset()

    
#     def __str__(self) -> str:
#         string = super().__str__()+'\n\t\t'
#         string += "affected by: "+ ', '.join([a.__str__() for a in self.axons])
#         return string




# class AbstractNeuromodulatoryWeightLRE(AbstractNeuromodulatoryLRE, AbstractWeightLRE):
#     def __init__(
#         self,
#         name: str = None,
#         **kwargs
#     ) -> None:
#         super().__init__(name, **kwargs)




# class RSTDP(AbstractNeuromodulatoryWeightLRE):
#     """
#     Reward-modulated Spike-Time Dependent Plasticity learning rule.

#     Implement the dynamics of RSTDP learning rule. You might need to implement\
#     different update rules based on type of connection.
#     """

#     def __init__(
#         self,
#         name: str = None,
#         stdp: STDP = STDP(),
#         tau: Union[float, torch.Tensor] = 1000.,
#         config_prohibit: bool = False,
#         **kwargs
#     ) -> None:
#         super().__init__(config_prohibit=True, name=name, **kwargs)
#         self.stdp = stdp
#         self.register_buffer("tau", torch.as_tensor(tau))
#         self.register_buffer("c", torch.as_tensor(0.))
#         self.config_prohibit = config_prohibit
#         self.config()

    
#     def config(self) -> bool:
#         if not super().config():
#             return False
#         self.stdp.set_name(self.name)
#         self.stdp.set_synapse(self.synapse)
#         self.stdp.set_dt(self.dt)
#         return True


#     def forward(self, direct_input: torch.Tensor = torch.as_tensor(0.)) -> None:
#         self.stdp.forward(direct_input=direct_input)
#         stdp_output = self.stdp.compute_updatings()
#         delta = (self.synapse.dendrite.spikes() + self.to_singlton_dendrite_shape(self.synapse.axon.spikes()))
#         self.c = self.c + stdp_output*delta - self.dt * self.c / self.tau
#         super().forward()


#     def compute_updatings(self) -> torch.Tensor:
#         dw = self.c * self.neuromodulators
#         return dw


#     def reset(self) -> None:
#         self.stdp.reset()
#         self.c.zero_()
#         super().reset()




# class FlatRSTDP(AbstractNeuromodulatoryWeightLRE):
#     def __init__(
#         self,
#         name: str = None,
#         stdp: STDP = FlatSTDP(),
#         config_prohibit: bool = False,
#         **kwargs
#     ) -> None:
#         super().__init__(config_prohibit=True, name=name, **kwargs)
#         self.stdp = stdp
#         self.config_prohibit = config_prohibit
#         self.config()

    
#     def config(self) -> bool:
#         if not super().config():
#             return False
#         self.stdp.set_name(self.name)
#         self.stdp.set_synapse(self.synapse)
#         self.stdp.set_dt(self.dt)
#         return True


#     def forward(self, direct_input: torch.Tensor = torch.as_tensor(0.)) -> None:
#         self.stdp.forward(direct_input=direct_input)
#         super().forward()


#     def compute_updatings(self) -> torch.Tensor:
#         dw = self.stdp.compute_updatings() * self.neuromodulators
#         return dw


#     def reset(self) -> None:
#         self.stdp.reset()
#         super().reset()