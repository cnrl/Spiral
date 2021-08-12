"""
Spiking soma add-on principles will apply some extra strategies on\
activity of a given spiking soma.
"""


import torch
from typing import Union, Callable
from typeguard import typechecked
from construction_requirements_integrator import construction_required
from add_on_class import AOC, covering_around
from ..spiking_soma import SpikingSoma




@typechecked
@covering_around([SpikingSoma])
class KWinnersTakeAllPrinciple(AOC):
    """
    Add-on class to add k-winners-take-all principle to a spiking soma.

    This module will modify output spikes of a giving spiking soma to perform\
    k-winners-take-all principle.
    Read more about add-on classes in add-on-class package documentation.

    Add-On Properties
    -----------------
    sharpness : torch.Tensor
        The sharpness of the depolarisation process.
    depolarization_threshold: torch.Tensor
        The membrane potential threshold of the depolarisation process in millivolts.

    Arguments
    ---------
    number_of_winners : int, Optional, default: 1
        Determines number of the winners (k) in k-winners-take-all principle.
    kwinners_take_all_principle_duration : float or torch.Tensor, Optional, default: 0.
        Determines the duration of k-winners-take-all principle in milliseconds.\
        The model ensures that there is a maximum of k winners in each duration.\
        If it set to a value less than `dt` of the core, The model will ensures\
        that there is k winners in each step.
    kwinners_take_all_spare_evaluation_criteria: Callables[SpikingSoma], Optional, default: lambda x: torch.rand(x.spike.shape)
        In case of spike synchronization, it determines the method of selecting the winners.
    """
    def __post_init__(
        self,
        number_of_winners: int = 1,
        kwinners_take_all_principle_duration: Union[float, torch.Tensor] = 0.,
        kwinners_take_all_spare_evaluation_criteria: Callable = None,
    ) -> None:
        self.number_of_winners = number_of_winners
        self.number_of_remaining_winners = 0
        self.register_buffer("kwinners_take_all_principle_duration", torch.as_tensor(kwinners_take_all_principle_duration))
        if self.kwinners_take_all_principle_duration.numel() > 1:
            raise Exception("kwinners_take_all_principle_duration must be a single float value.")
        self.register_buffer("remaining_kwinners_take_all_principle_duration", torch.as_tensor(0.))
        self.kwinners_take_all_spare_evaluation_criteria = (lambda x: 0)\
            if kwinners_take_all_spare_evaluation_criteria is None else\
            kwinners_take_all_spare_evaluation_criteria


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
        self.__core._fire_axon_hillock(self, clamps=clamps, unclamps=unclamps)
        
        if self.remaining_kwinners_take_all_principle_duration<self.dt:
            self.remaining_kwinners_take_all_principle_duration = self.kwinners_take_all_principle_duration
            self.number_of_remaining_winners = self.number_of_winners

        if self.spike.sum()>self.number_of_remaining_winners:
            new_spikes = torch.zeros(self.spike.numel(), dtype=bool)
            new_spikes[
                (self.kwinners_take_all_spare_evaluation_criteria(self) - float('inf')*(~self.spike))\
                    .reshape(-1).topk(self.number_of_remaining_winners).indices\
            ] = True
            self._spike = new_spikes.reshape(self.spike.shape)
        self.number_of_remaining_winners -= self.spike.sum()
        self.remaining_kwinners_take_all_principle_duration -= self.dt
        

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
        self.remaining_kwinners_take_all_principle_duration.zero_()
        self.__core.reset(self)