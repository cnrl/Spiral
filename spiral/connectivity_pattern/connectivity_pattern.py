"""
"""


import torch
from typing import Union, Iterable
from typeguard import typechecked
from abc import ABC, abstractmethod
from constant_properties_protector import CPP
from construction_requirements_integrator import CRI, construction_required




@typechecked
class ConnectivityPattern(torch.nn.Module, CRI, ABC):
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
    def _add_batch_dims(
        self,
        connectivity: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.zeros(self.batch, *self.source, self.batch, *self.target)
        for i in range(self.batch):
            output[i, :, i, :] = connectivity
        return output


    @abstractmethod
    @construction_required
    def __call__(
        self,
    ) -> torch.Tensor:
        pass


    def reset(
        self
    ) -> None:
        pass




@typechecked
class NotConnectivity(ConnectivityPattern):
    def __init__(
        self,
        connectivity_pattern: ConnectivityPattern,
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
        self.connectivity_pattern = connectivity_pattern


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
        self.connectivity_pattern.meet_requirement(source=source)
        self.connectivity_pattern.meet_requirement(target=target)
        self.connectivity_pattern.meet_requirement(batch=batch)
        self.connectivity_pattern.meet_requirement(dt=dt)


    @construction_required
    def __call__(
        self,
    ) -> torch.Tensor:
        return ~(self.connectivity_pattern()).bool()


    def reset(
        self
    ) -> None:
        self.connectivity_pattern.reset()




@typechecked
class AggConnectivity(ConnectivityPattern, ABC):
    def __init__(
        self,
        connectivity_patterns: Iterable[ConnectivityPattern],
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
        self.connectivity_patterns = connectivity_patterns


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
        for connectivity_pattern in self.connectivity_patterns:
            connectivity_pattern.meet_requirement(source=source)
            connectivity_pattern.meet_requirement(target=target)
            connectivity_pattern.meet_requirement(batch=batch)
            connectivity_pattern.meet_requirement(dt=dt)


    @abstractmethod
    @construction_required
    def __call__(
        self,
    ) -> torch.Tensor:
        pass


    def reset(
        self
    ) -> None:
        for connectivity_pattern in self.connectivity_patterns:
            connectivity_pattern.reset()




@typechecked
class AndConnectivity(AggConnectivity):
    @construction_required
    def __call__(
        self,
    ) -> torch.Tensor:
        pattern = torch.as_tensor(True)
        for connectivity_pattern in self.connectivity_patterns:
            pattern = pattern * connectivity_pattern()
        return pattern.bool()




@typechecked
class OrConnectivity(AggConnectivity):
    @construction_required
    def __call__(
        self,
    ) -> torch.Tensor:
        pattern = torch.as_tensor(True)
        for connectivity_pattern in self.connectivity_patterns:
            pattern = pattern + connectivity_pattern()
        return pattern.bool()




@typechecked
class FixedConnectivity(ConnectivityPattern, ABC):
    def __init__(
        self,
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
        self.is_constructed = True
        self.connectivity = self._generate_connectivity()
        self.connectivity = self._add_batch_dims(connectivity=self.connectivity)


    @abstractmethod
    @construction_required
    def _generate_connectivity(
        self
    ) -> torch.Tensor:
        pass


    @construction_required
    def __call__(
        self,
    ) -> torch.Tensor:
        return self.connectivity




@typechecked
class AutapseConnectivity(FixedConnectivity):
    def __construct__(
        self,
        source: Iterable[int],
        target: Iterable[int],
        batch: int,
        dt: Union[float, torch.Tensor],
    ) -> None:
        if source!=target:
            raise Exception(f"Can not build an autapse connectivity pattern between two different shapes: {source} != {target}")
        super().__construct__(
            source=source,
            target=target,
            batch=batch,
            dt=dt,
        )


    @construction_required
    def _generate_connectivity(
        self,
    ) -> torch.Tensor:
        return torch.diag(torch.ones(torch.prod(torch.as_tensor(self.source)))).reshape(*self.source, *self.target).bool()




@typechecked
class RandomConnectivity(FixedConnectivity):
    def __init__(
        self,
        rate: Union[float, torch.Tensor],
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
        self.register_buffer('rate', torch.as_tensor(rate))
        if self.rate.numel()!=1:
            raise Exception("Rate must be a single float value.")
        self.set_construction_permission(construction_permission)


    @construction_required
    def _generate_connectivity(
        self
    ) -> torch.Tensor:
        return torch.rand(*self.source, *self.target).uniform_() > self.rate




@typechecked
class RandomFixedCouplingConnectivity(RandomConnectivity):
    @construction_required
    def _generate_connectivity(
        self,
    ) -> torch.Tensor:
        count = int(torch.prod(torch.as_tensor(*self.source, *self.target))*self.rate)
        output = torch.rand(*self.source, *self.target)
        threshold = output.reshape(-1).sort()[0][-count]
        return (output >= threshold)




@typechecked
class RandomFixedPresynapticPartnersConnectivity(RandomConnectivity):
    @construction_required
    def _generate_connectivity(
        self,
    ) -> torch.Tensor:
        count = int(torch.prod(torch.as_tensor(self.source))*self.rate)
        output = torch.rand(*self.source, *self.target)
        flatted = output.reshape(-1, *self.target)
        threshold = torch.topk(flatted, flatted.shape[0], dim=0, largest=False)[0][-count]
        return (output >= threshold)
