"""
Spiking soma add-on principles will apply some extra strategies on\
activity of a given spiking soma.
"""


import torch
from typing import Union, Callable, Iterable
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
    k-winners-take-all principle.\
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
        self.number_of_remaining_winners = []
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
            self.number_of_remaining_winners = [self.number_of_winners]*self.batch

        for b in range(self.batch):
            if self.spike[b].sum()>self.number_of_remaining_winners[b]:
                new_spikes = torch.zeros_like(self.spike[b]).reshape(-1)
                new_spikes[
                    (self.kwinners_take_all_spare_evaluation_criteria(self)[b] - float('inf')*(~self.spike[b]))\
                        .reshape(-1).topk(self.number_of_remaining_winners[b]).indices\
                ] = True
                self._spike[b] = new_spikes.reshape(self.spike[b].shape)
            self.number_of_remaining_winners[b] -= self.spike[b].sum()
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




@typechecked
@covering_around([SpikingSoma])
class KRandomClampsPrinciple(AOC):
    """
    Add-on class to add k-clamps principle to a spiking soma.

    This module will modify output spikes of a giving spiking soma to perform\
    k-clamps principle.\
    Read more about add-on classes in add-on-class package documentation.

    Add-On Properties
    -----------------
    clamps_distribution : Callable[SpikingSoma]
        A function that returns distribution of selecting clamps.

    Arguments
    ---------
    clamps_distribution : Callable[SpikingSoma], Optional, default: lambda x: torch.ones_like(x.spike)/x.spike.numel().
        A function that returns distribution of selecting clamps.
    """
    def __post_init__(
        self,
        clamps_distribution: Callable = None,
    ) -> None:
        if clamps_distribution is None:
            clamps_distribution = lambda x: torch.ones_like(x.spike).float()
        self.clamps_distribution = clamps_distribution


    @construction_required
    @analysis_point
    def progress(
        self,
        direct_input: torch.Tensor = torch.as_tensor(0.),
        clamps: torch.Tensor = torch.as_tensor(False),
        unclamps: torch.Tensor = torch.as_tensor(False),
        k_clamps: int = 0,
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
        k_clamps : int, Optional, default: 0
            Determines number of clamps producing by k-clamps principle in this step.
        
        Returns
        -------
        None
        
        """
        self.__core.progress(
            self,
            direct_input=direct_input,
            clamps=clamps+\
                torch.zeros_like(self.spike)\
                    .reshape(self.batch, -1)\
                    .scatter(
                        1,
                        self.clamps_distribution(self)\
                            .reshape(self.batch, -1)\
                            .multinomial(k_clamps),
                        True,
                    ).reshape(self.spike.shape),
            unclamps=unclamps,
        )