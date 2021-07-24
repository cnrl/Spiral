"""
This is an auxiliary class for generating semi-random current inputs.
"""


import torch
from typing import Iterable, Dict
from typeguard import typechecked




@typechecked
class FunctionGenerator:
    """
    The class for generating semi-random current inputs.\
    Just use generate function without instancing: `FunctionGenerator.generate(...)`.
    """

    @staticmethod
    def __get_steps(
        length: int,
        moves: Dict[int, float],
    ) -> zip:
        """
        Translates input dictionaries to segments as steps.
        
        Arguments
        ---------
        length : int
            The total length of asked function.
        moves: Dict[int, float]
            The input dictionary.
        
        Returns
        -------
        steps : zip[int, int]
            A zip of start and end of the steps.
        
        """
        points = list(moves)+[length]
        return zip(points[:-1], points[1:])
        
    @classmethod
    def __generate_baseline(
        fg,
        length: int,
        baseline: Dict[int, float],
    ) -> torch.Tensor:
        """
        Generates baseline of the asked function.
        
        Arguments
        ---------
        length : int
            The total length of asked function.
        baseline: Dict[int, float]
            The input dictionary.
        
        Returns
        -------
        baseline : torch.Tensor
            The baseline of the asked function.
        
        """
        output = torch.zeros(length)
        for left,right in fg.__get_steps(length, baseline):
            v = baseline[left]
            output[left:right] += v
        return output

    @classmethod
    def __generate_slope(
        fg,
        length: int,
        slope: Dict[int, float],
    ) -> torch.Tensor:
        """
        Generates slopes of the asked function.
        
        Arguments
        ---------
        length : int
            The total length of asked function.
        slope: Dict[int, float]
            The input dictionary.
        
        Returns
        -------
        slope : torch.Tensor
            The slopes of the asked function.
        
        """
        output = torch.zeros(length)
        for left,right in fg.__get_steps(length, slope):
            v = slope[left]
            if v!=0:
                output[left:right] += torch.arange(0, (right-left)*v, v)
            output[right:] = output[right-1]+v
        return output
    
    @classmethod
    def __generate_noise(
        fg,
        length: int,
        noise: Dict[int, float],
        shape: Iterable[int] = (),
    ) -> torch.Tensor:
        """
        Generates noises for the asked function.
        
        Arguments
        ---------
        length : int
            The total length of asked function.
        noise: Dict[int, float]
            The input dictionary.
        shape: Iterable[int], Optional, default: ()
            To generate inter population noises.
        
        Returns
        -------
        noise : torch.Tensor
            The generated noise.
        
        """
        output = torch.zeros((length, *shape))
        for left,right in fg.__get_steps(length, noise):
            v = noise[left]
            if v!=0:
                output[left:right] += torch.normal(.0, torch.as_tensor(float(v)), (right-left,*shape))
        return output
    
    
    @staticmethod
    def __add_singletons_to_shape(
        tensor: torch.Tensor,
        shape: Iterable[int] = (),
    ) -> torch.Tensor:
        """
        An auxiliary function to convert a function to a demographic function.
        
        Arguments
        ---------
        tensor : torch.Tensor
            The function.
        shape: Iterable[int], Optional, default: ()
            Shape of the population.
        
        Returns
        -------
        function : torch.Tensor
            The demographic function.
        
        """
        return tensor.reshape(tensor.shape[0], *[1 for i in shape])
    

    @classmethod
    def generate(
        fg,
        length: int,
        shape: Iterable[int] = (),
        baseline: Dict[int, float] = {0: 0.},
        slope: Dict[int, float] = {0: 0.},
        noise: Dict[int, float] = {0: 0.},
        population_noise: Dict[int, float] = {0: 0.},
    ) -> torch.Tensor:
        """
        The main function to generate asked semi-random function.
        
        Arguments
        ---------
        length : int, Necessary
            The length of asked function.
        shape: Iterable[int], Optional, default: ()
            The shape of asked demographic population.
        baseline: Dict[int, float], Optional, default: {0: 0.}
            Asked baseline of the function.
        slope: Dict[int, float], Optional, default: {0: 0.}
            Asked slope of the function.
        noise: Dict[int, float], Optional, default: {0: 0.}
            Asked noise of the function.
        population_noise: Dict[int, float], Optional, default: {0: 0.}
            Asked inter population noise of the demographic function.
        
        Returns
        -------
        function : torch.Tensor
            The generated function.
        
        """
        return (
            fg.__add_singletons_to_shape(
                (
                    fg.__generate_baseline(length, baseline) +\
                    fg.__generate_slope(length, slope) +\
                    fg.__generate_noise(length, noise)
                ),
                shape
            ) +\
            fg.__generate_noise(length, population_noise, shape)
        )