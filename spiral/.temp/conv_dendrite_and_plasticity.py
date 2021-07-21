class FilteringDendriteSet2D(Dendrite):
    def __init__(
        self,
        name: str = None,
        filt: AbstractFilter = None,
        config_prohibit: bool = False,
        **kwargs
    ) -> None:
        super().__init__(
            name=name,
            config_prohibit=True,
            **kwargs
        )
        self.config_prohibit = config_prohibit
        self.set_filter(filt)


    def config_permit(self):
        return (super().config_permit() and (self.filter is not None))

    
    def config(self) -> bool:
        if not self.config_permit():
            return False
        assert self.required_population_shape()==self.population_shape, "terminal shape doesn't match with population shape according to filter"
        return super().config()


    def set_filter(self, filt) -> bool:
        if self.configed:
            return False
        self.filter = filt
        if self.filter is not None:
            self.add_module('filter', self.filter)
        self.config()
        return True


    def required_population_shape(self) -> Iterable[int]:
        assert (self.terminal_shape is not None and self.filter is not None), \
            "please set terminal and filter at the first place."
        return self.filter(torch.zeros(self.terminal_shape)).shape


    def forward(self, neurotransmitters: torch.Tensor) -> None: #doesn't replace nan values
        self.I = self.filter(neurotransmitters)


    def currents(self) -> torch.Tensor:
        return self.I



class AbstractKernelWeightLRE(CombinableLRE):
    def __init__(
        self,
        name: str = None,
        **kwargs
    ) -> None:
        super().__init__(name=name, **kwargs)


    @abstractmethod
    def compute_updatings(self) -> torch.Tensor: # output = dw
        pass


    def update(self, dw: torch.Tensor) -> None:
        w = self.synapse.dendrite.filter.core.weight.data
        wmin = self.synapse.dendrite.wmin
        wmax = self.synapse.dendrite.wmax
        w += dw
        w[w<wmin] = wmin
        w[w>wmax] = wmax
        self.synapse.dendrite.filter.core.weight.data = w




class KernelSTDP(AbstractKernelWeightLRE):
    def __init__(
        self,
        name: str = None,
        dims: int = 2,
        pre_traces: AbstractSynapticTagger = None,
        post_traces: AbstractSynapticTagger = None,
        ltp_wdlr: Callable = stdp_wdlr(.1), #LTP weight dependent learning rate
        ltd_wdlr: Callable = stdp_wdlr(.1), #LTD weight dependent learning rate
        config_prohibit: bool = False,
        **kwargs
    ) -> None:  
        super().__init__(config_prohibit=True, name=name, **kwargs)
        self.dims = dims
        self.pre_traces = pre_traces if pre_traces is not None else STDPST()
        self.post_traces = post_traces if post_traces is not None else STDPST()
        self.ltp_wdlr = ltp_wdlr
        self.ltd_wdlr = ltd_wdlr
        self.config_prohibit = config_prohibit
        self.config()

    
    def config(self) -> bool:
        if not super().config():
            return False
        self.pre_traces.set_shape(self.synapse.axon.population_shape)
        self.post_traces.set_shape(self.synapse.dendrite.population_shape)
        self.pre_traces.set_dt(self.dt)
        self.post_traces.set_dt(self.dt)
        return True


    def compute_lrs(self) -> tuple: # ltp_lr,ltd_lr
        w = self.synapse.dendrite.filter.core.weight.data
        wmin = self.synapse.dendrite.wmin
        wmax = self.synapse.dendrite.wmax
        ltp_lr = self.ltp_wdlr(w, wmin, wmax)
        ltd_lr = self.ltd_wdlr(w, wmin, wmax)
        return ltp_lr,ltd_lr


    def presynaptic_process(self, tensor): ##### it need a better name!!!!
        if not self.synapse.dendrite.filter.channel_inputing:
            tensor = tensor.unsqueeze(0)
        kernel_size = self.synapse.dendrite.filter.core.weight.data.shape[-len(tensor.shape)+1:]
        stride = self.synapse.dendrite.filter.core.stride
        padding = self.synapse.dendrite.filter.core.padding
        for i,pad in enumerate(padding):
            shape = list(tensor.shape)
            shape[i+1] = pad
            tensor = torch.cat([torch.zeros(shape),tensor,torch.zeros(shape)], axis=i+1)
        for i,strd in enumerate(stride):
            tensor = tensor.unfold(i+1,kernel_size[i],strd)
        tensor = tensor.unsqueeze(0)
        return tensor


    def postsynaptic_process(self, tensor): ##### it need a better name!!!!
        if not self.synapse.dendrite.filter.channel_outputing:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.reshape(tensor.shape[0], 1, *tensor.shape[1:])
        tensor = tensor.reshape(*tensor.shape, *[1]*(len(tensor.shape)-2))
        return tensor

    
    def compute_updatings(self) -> torch.Tensor:
        ltp_lr,ltd_lr = self.compute_lrs()
        
        ltp = self.presynaptic_process(self.pre_traces.traces()) * self.postsynaptic_process(self.synapse.dendrite.spikes())
        ltp = ltp_lr * ltp.sum(axis=list(range(2,len(ltp_lr.shape))))

        ltd = self.postsynaptic_process(self.post_traces.traces()) * self.presynaptic_process(self.synapse.axon.spikes())
        ltd = ltd_lr * ltd.sum(axis=list(range(2,len(ltd_lr.shape))))
        
        dw = self.dt * (ltp - ltd)
        return dw


    def forward(self, direct_input: torch.Tensor = torch.as_tensor(0.)) -> None:
        self.pre_traces.forward(self.synapse.axon.spikes())
        self.post_traces.forward(self.synapse.dendrite.spikes())
        super().forward()


    def reset(self) -> None:
        self.pre_traces.reset()
        self.post_traces.reset()
        super().reset()
