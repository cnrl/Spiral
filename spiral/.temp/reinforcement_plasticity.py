class AbstractNeuromodulatoryLRE(CombinableLRE):
    def __init__(
        self,
        name: str = None,
        **kwargs
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.axons = {}
        self.register_buffer("neuromodulators", torch.as_tensor(0.))
            

    def add_axon(self, axon_set: Union[AbstractAxonSet, Iterable]) -> None:
        if hasattr(axon_set, '__iter__'):
            for o in axon_set:
                self.usadd_axone(o)
        else:
            self.axons[axon_set.name] = axon_set


    def remove_axon(self, name: str) -> None:
        del self.axons[name]


    def collect_neuromodulators(self, direct_input: torch.Tensor = torch.as_tensor(0.)):
        neuromodulators = direct_input
        for axon_set in self.axons.values():
            neuromodulators = neuromodulators + axon_set.neurotransmitters()
        return neuromodulators


    def forward(self, direct_input: torch.Tensor = torch.as_tensor(0.)) -> None:
        self.neuromodulators = self.collect_neuromodulators(direct_input=direct_input)


    def reset(self) -> None:
        self.neuromodulators.zero_()
        super().reset()

    
    def __str__(self) -> str:
        string = super().__str__()+'\n\t\t'
        string += "affected by: "+ ', '.join([a.__str__() for a in self.axons])
        return string




class AbstractNeuromodulatoryWeightLRE(AbstractNeuromodulatoryLRE, AbstractWeightLRE):
    def __init__(
        self,
        name: str = None,
        **kwargs
    ) -> None:
        super().__init__(name, **kwargs)




class RSTDP(AbstractNeuromodulatoryWeightLRE):
    """
    Reward-modulated Spike-Time Dependent Plasticity learning rule.

    Implement the dynamics of RSTDP learning rule. You might need to implement\
    different update rules based on type of connection.
    """

    def __init__(
        self,
        name: str = None,
        stdp: STDP = STDP(),
        tau: Union[float, torch.Tensor] = 1000.,
        config_prohibit: bool = False,
        **kwargs
    ) -> None:
        super().__init__(config_prohibit=True, name=name, **kwargs)
        self.stdp = stdp
        self.register_buffer("tau", torch.as_tensor(tau))
        self.register_buffer("c", torch.as_tensor(0.))
        self.config_prohibit = config_prohibit
        self.config()

    
    def config(self) -> bool:
        if not super().config():
            return False
        self.stdp.set_name(self.name)
        self.stdp.set_synapse(self.synapse)
        self.stdp.set_dt(self.dt)
        return True


    def forward(self, direct_input: torch.Tensor = torch.as_tensor(0.)) -> None:
        self.stdp.forward(direct_input=direct_input)
        stdp_output = self.stdp.compute_updatings()
        delta = (self.synapse.dendrite.spikes() + self.to_singlton_dendrite_shape(self.synapse.axon.spikes()))
        self.c = self.c + stdp_output*delta - self.dt * self.c / self.tau
        super().forward()


    def compute_updatings(self) -> torch.Tensor:
        dw = self.c * self.neuromodulators
        return dw


    def reset(self) -> None:
        self.stdp.reset()
        self.c.zero_()
        super().reset()




class FlatRSTDP(AbstractNeuromodulatoryWeightLRE):
    def __init__(
        self,
        name: str = None,
        stdp: STDP = FlatSTDP(),
        config_prohibit: bool = False,
        **kwargs
    ) -> None:
        super().__init__(config_prohibit=True, name=name, **kwargs)
        self.stdp = stdp
        self.config_prohibit = config_prohibit
        self.config()

    
    def config(self) -> bool:
        if not super().config():
            return False
        self.stdp.set_name(self.name)
        self.stdp.set_synapse(self.synapse)
        self.stdp.set_dt(self.dt)
        return True


    def forward(self, direct_input: torch.Tensor = torch.as_tensor(0.)) -> None:
        self.stdp.forward(direct_input=direct_input)
        super().forward()


    def compute_updatings(self) -> torch.Tensor:
        dw = self.stdp.compute_updatings() * self.neuromodulators
        return dw


    def reset(self) -> None:
        self.stdp.reset()
        super().reset()