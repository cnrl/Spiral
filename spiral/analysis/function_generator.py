import torch
from typing import Iterable, Dict

class FunctionGenerator:
    @staticmethod
    def __get_steps(
        length: int,
        moves: Dict[int, float],
    ) -> zip:
        points = list(moves)+[length]
        return zip(points[:-1], points[1:])
        
    @classmethod
    def __generate_baseline(
        fg,
        length: int,
        baseline: Dict[int, float],
    ) -> torch.Tensor:
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
        output = torch.zeros(length)
        for left,right in fg.__get_steps(length, slope):
            v = slope[left]
            if v!=0:
                output[left:right+1] += torch.arange(0, (right+1-left)*v, v)
            if right<length:
                output[right+1:] = output[right]
        return output
    
    @classmethod
    def __generate_noise(
        fg,
        length: int,
        noise: Dict[int, float],
        shape: Iterable[int] = (),
    ) -> torch.Tensor:
        output = torch.zeros((length, *shape))
        for left,right in fg.__get_steps(length, noise):
            v = noise[left]
            if v!=0:
                output[left:right] += torch.normal(.0, torch.tensor(float(v)), (right-left,*shape))
        return output
    
    
    @staticmethod
    def __add_singletons_to_shape(
        tensor: torch.Tensor,
        shape: Iterable[int] = (),
    ) -> torch.Tensor:
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
    ):
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