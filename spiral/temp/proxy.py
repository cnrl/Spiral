class AbstractPopulationProxy(AbstractNeuralPopulation):
    def __init__(
        self,
        population: AbstractNeuralPopulation,
        name: str = None,
        dt: float = None,
        **kwargs
    ) -> None:
        super().__init__(
            name=name if name is not None else f'Proxy[{population.name}]',
            shape=population.shape,
            **kwargs
        )
        self.population = population
        self.set_dt(dt if dt is not None else population.dt)


    def set_dt(self, dt:float):
        if 'population' in self.__dict__['_modules']:
            self.population.set_dt(dt)
        super().set_dt(dt)
            

    def backward(self) -> None:
        self.population.backward()
        super().backward()


    def forward(self,
            direct_input: torch.Tensor = torch.tensor(0.),
            clamps: torch.Tensor = torch.tensor(False),
            unclamps: torch.Tensor = torch.tensor(False)) -> None:
        self.population.forward(direct_input=self.collect_I(direct_input), clamps=clamps, unclamps=unclamps)
        self.compute_spike(self.population.spikes())
        spikes = self.spikes()
        for axon_set in self.axons.values():
            axon_set.forward(spikes)

    
    @abstractmethod
    def compute_spike(self, plain_spikes: torch.Tensor,
            clamps: torch.Tensor = torch.tensor(False),
            unclamps: torch.Tensor = torch.tensor(False)) -> None:
        super().compute_spike(clamps, unclamps)


    def reset(self) -> None:
        self.population.reset()
        super().reset()




class DisposablePopulationProxy(AbstractPopulationProxy):
    def __init__(
        self,
        population: AbstractNeuralPopulation,
        **kwargs
    ) -> None:
        super().__init__(
            population=population,
            **kwargs
        )
        self.register_buffer("consumed", torch.zeros(self.shape, dtype=torch.bool))


    def compute_spike(self,
            plain_spikes: torch.Tensor,
            clamps: torch.Tensor = torch.tensor(False),
            unclamps: torch.Tensor = torch.tensor(False)) -> None:
        self.s = plain_spikes * ~self.consumed
        super().compute_spike(plain_spikes, clamps, unclamps)
        self.consumed += self.s


    def reset(self) -> None:
        super().reset()
        self.consumed.zero_()




class KWinnerTakeAllPopulationProxy(AbstractPopulationProxy):
    def __init__(
        self,
        population: AbstractNeuralPopulation,
        lateral: Iterable[int],
        k: int = 1,
        critical_comparable: Callable = lambda x: x.u,
        **kwargs
    ) -> None:
        super().__init__(
            population=population,
            **kwargs
        )
        self.lateral = lateral
        self.register_buffer("ban_neurons", torch.zeros(self.shape, dtype=torch.bool))
        self.k = k
        self.k_need = k
        self.critical_comparable = critical_comparable
        self.features_dim = len(self.shape)-len(self.lateral)


    def compute_spike(self,
            plain_spikes: torch.Tensor,
            clamps: torch.Tensor = torch.tensor(False),
            unclamps: torch.Tensor = torch.tensor(False)) -> None:
        s = plain_spikes.clone()
        comparable = self.critical_comparable(self.population)
        self.s.zero_()
        while self.k_need>0:
            s *= ~self.ban_neurons
            if not torch.any(s):
                break
            valid_comparable = comparable.clone()
            valid_comparable[~s] = -float('inf')
            this_spike = torch.zeros_like(s).reshape(-1)
            this_spike[valid_comparable.reshape(-1).topk(1).indices] = 1
            this_spike = this_spike.reshape(s.shape)
            self.s += this_spike.bool()
            self.ban_neurons[[l[0] for l in torch.where(this_spike)[:self.features_dim]]] = True
            if 0 not in self.lateral:
                self.ban_neurons[[slice(None, None, None) for i in range(self.features_dim)]+\
                                [slice(max(l[0]-self.lateral[i]//2, 0), l[0]+self.lateral[i]//2+1, None) \
                                for i,l in enumerate(torch.where(this_spike)[self.features_dim:])]] = True
            self.k_need -= 1
        super().compute_spike(plain_spikes, clamps, unclamps)
        

    def reset(self) -> None:
        super().reset()
        self.k_need = self.k
        self.ban_neurons.zero_()