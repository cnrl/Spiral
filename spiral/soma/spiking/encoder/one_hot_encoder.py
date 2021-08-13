"""
TODO
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
class OneHotEncoder(SpikingSoma):
    """
    TODO
    """
    def __init__(
        self,
        name: str,
        shape: Iterable[int] = (),
        size: int = None,
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
        CPP.protect(self, 'size')
        self.add_to_construction_requirements(size=size)
        self.set_construction_permission(construction_permission)
    

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
        self._size = size
        self._input_shape = shape
        self.register_buffer('state', torch.zeros(self.batch, *self._input_shape)-1)


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
            raise Exception("Can not attach a dendrite to one-hot encoder.")

        return super().use(organ)


    def _integrate_inputs(
        self,
        direct_input: torch.Tensor = torch.as_tensor(0.)
    ) -> torch.Tensor:
        """
        Calculates the sum of currents from dendrites or direct inputs.
        
        Arguments
        ---------
        direct_input : torch.Tensor
            Direct current input in milliamperes.
        
        Returns
        -------
        total_input_current : torch.Tensor
            The sum of currents from dendrites or direct inputs in milliamperes.
        
        """
        i = torch.zeros(self.batch, *self._input_shape, device=self.dt.device)
        i += direct_input.to(self.dt.device)
        return i


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
        self.state = inputs.to(torch.int64)
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
        embpties = (self.state==-1)
        self.state[embpties] = 0
        self._spike = torch.nn.functional.one_hot(self.state, num_classes=self.size).bool()
        self._spike[embpties] = False
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
        self.state -= 1
        super().reset()




@typechecked
@covering_around([OneHotEncoder])
class Object2IndexReceiver(AOC):
    """
    TODO

    Arguments
    ---------
    sharpness : float or torch.Tensor, Optional, default: 2.0
        Determines the sharpness of the depolarisation process.
    depolarization_threshold : float or torch.Tensor, Optional, default: -50.4
        Determines the membrane potential threshold of the depolarisation process in millivolts.
    """
    def __post_init__(
        self,
        objects: Union[Dict, Iterable],
        default: int = -1,
        unknown_exception: bool = True,
    ) -> None:
        if not hasattr(objects, '__getitem__'):
            objects = {o:i for i,o in enumerate(objects)}
        self.object2index = objects
        self.__default = default
        self.__unknown_exception = unknown_exception
        if not self.is_constructed:
            self.meet_requirement(size=max(self.object2index.values())+1)


    @construction_required
    def progress(
        self,
        direct_input: Any,
        clamps: torch.Tensor = torch.as_tensor(False),
        unclamps: torch.Tensor = torch.as_tensor(False)
    ) -> None:
        """
        Simulate the soma activity for a single step.

        Arguments
        ---------
        inputs : torch.Tensor
            Input current in milliamperes.
        clamps : torch.Tensor[bool], Optional, default: torch.tensor(False)
            Forcing neurons to fire.
        unclamps : torch.Tensor[bool], Optional, default: torch.tensor(False)
            Forcing neurons not to fire.
        
        Returns
        -------
        None
        
        """
        if len(self._input_shape)>0:
            for _ in range(len(self._input_shape)):
                direct_input = [sub_sub_input for sub_input in direct_input for sub_sub_input in sub_input]
        if self.__unknown_exception:
            direct_input = [self.object2index[i] for i in direct_input]
        else:
            direct_input = [self.object2index.get(i, self.__default) for i in direct_input]
        direct_input = torch.as_tensor(direct_input).reshape(self.batch, *self._input_shape)
        self.__core.progress(
            self,
            direct_input=direct_input,
            clamps=clamps,
            unclamps=unclamps,
        )