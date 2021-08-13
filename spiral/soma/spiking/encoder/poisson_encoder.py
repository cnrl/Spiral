"""
TODO
needs cleaning
"""


from __future__ import annotations
import torch
from typing import Union, Dict, Iterable, Any
from typeguard import typechecked
from constant_properties_protector import CPP
from construction_requirements_integrator import construction_required
from add_on_class import AOC, covering_around
from spiral.analysis import Analyzer, analytics
from ..spiking_soma import SpikingSoma
from spiral.axon.axon import Axon
from spiral.dendrite.dendrite import Dendrite




@typechecked
class PoissonEncoder(SpikingSoma):
    """
    TODO
    """
    def __init__(
        self,
        name: str,
        rate: Union[int, torch.Tensor],
        shape: Iterable[int] = None,
        batch: int = None,
        dt: Union[float, torch.Tensor] = None,
        construction_permission: bool = True,
    ) -> None:
        super().__init__(
            name=name,
            shape=shape,
            batch=batch,
            dt=dt,
            construction_permission=False,
        )
        self.register_buffer("rate", torch.as_tensor(rate))
    

    def __construct__(
        self,
        shape: Iterable[int],
        batch: int,
        dt: Union[float, torch.Tensor],
        size: int,
    ) -> None:
        """
        This function is related to the completion of the construction process.\
        Read more about `__construct__` in construction-requirements-integrator package documentation.
        
        Arguments
        ---------
        shape : Iterable of int
            Defines the topology of somas in the population.
        batch : int
            Determines the batch size.
        dt : float or torch.Tensor
            Time step in milliseconds.
        
        Returns
        -------
        None
        
        """
        super().__construct__(
            shape=(*shape, size),
            batch=batch,
            dt=dt
        )
        self.register_buffer('state', torch.zeros(self.batch, *self.shape))


    def use(
        self,
        organ: Union[Axon, Dendrite]
    ) -> OneHotEncoder:
        """
        Attaches an organ to the soma.
        
        Arguments
        ---------
        organ : Axon or Dendrite
            Attaching organ.
        
        Returns
        -------
        self: Soma
            With the aim of making chains possible.
        
        """
        if issubclass(type(organ), Dendrite):
            raise Exception("Can not attach a dendrite to poisson encoder.")

        return super().use(organ)


    def _process(
        self,
        inputs: torch.Tensor,
    ) -> None:
        """
        Calculates the dynamics of neurons.

        Arguments
        ---------
        inputs : torch.Tensor
            Input current in milliamperes.

        Returns
        -------
        None
        
        """
        self.state = inputs * self.rate
        super()._process(inputs=inputs)


    def _fire_axon_hillock(
        self,
        clamps: torch.Tensor = torch.as_tensor(False),
        unclamps: torch.Tensor = torch.as_tensor(False),
    ) -> None:
        """
        Compute spikes.\
        This process simulates the process of firing that occurs in axon hillock part of the neuron body.        
        
        Arguments
        ---------
        clamps : torch.Tensor[bool], Optional, default: torch.tensor(False)
            Forcing neurons to fire.
        unclamps : torch.Tensor[bool], Optional, default: torch.tensor(False)
            Forcing neurons not to fire.

        Returns
        -------
        None
        
        """
        self._spike = torch.bernoulli(self.state).type(torch.bool)
        super()._fire_axon_hillock(clamps=clamps, unclamps=unclamps)


    @construction_required
    def reset(
        self,
    ) -> None:
        """
        Refractor and reset the somas and connected organs.
        
        Returns
        -------
        None
        
        """
        self.state.zero_()
        super().reset()