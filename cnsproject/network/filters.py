from abc import abstractmethod
from typing import Union, Iterable
from .kernels import DoG_kernel, gabor_kernel
import torch


class AbstractFilter(torch.nn.Module):
    def __init__(
        self,
        transform=lambda x: x
    ) -> None:
        super().__init__()
        self.transform = transform


    @abstractmethod
    def __call__(self, data) -> torch.Tensor:
        return self.transform(data)

    


class IneffectiveFilter(AbstractFilter):
    def __init__(
        self,
    ) -> None:
        super().__init__()


    def __call__(self, data) -> torch.Tensor:
        return super().__call__(data)




class KernelConvFilter(AbstractFilter):
    def __init__(
        self,
        kernel, #shape=(1,c,x,x) as (batch,channel,size,size)
        transform=lambda x: x,
        **kwargs
    ) -> None:
        super().__init__(transform=transform)
        b,c,x,y = kernel.shape
        assert x==y, "kernel shape must be square"
        # self.filter = torch.nn.Conv2d(
        #     in_channels=c,
        #     out_channels=c,
        #     kernel_size=x,
        #     groups=c,
        #     **kwargs
        # )
        # self.filter.weight.data = kernel
        # self.filter.weight.requires_grad = False
        # self.add_module('filter', self.filter)
        from ..utils import conv2d
        self.filter = conv2d(kernel=kernel[0][0], **kwargs)


    def __call__(self, data) -> torch.Tensor: # data in shape=(b,c,x,y) as (batch,channels,height,width)
        data = super().__call__(data)
        if len(data.shape)==3:
            data = data.unsqueeze_(0)
        return self.filter(data).detach()




class DoGFilter(KernelConvFilter):
    def __init__(
        self,
        kernel_size: int = 3,
        std1: Union[float,torch.Tensor] = 1.,
        std2: Union[float,torch.Tensor] = 2.,
        channels: int = 1,
        off_center=False,
        **kwargs
    ) -> None:
        super().__init__(
            kernel=DoG_kernel(
                kernel_size=kernel_size,
                std1=std1,
                std2=std2,
                channels=channels,
                off_center=off_center
            ),
            **kwargs
        )




class GaborFilter(KernelConvFilter):
    def __init__(
        self,
        kernel_size: int = 3,
        wavelength: Union[float,torch.Tensor] = 1.,
        orientation: torch.Tensor = torch.tensor(0.),
        std: Union[float,torch.Tensor] = 1.,
        aspect_ratio: Union[float,torch.Tensor] = 1.,
        channels: int = 1,
        off_center=False,
        **kwargs
    ) -> None:
        super().__init__(
            kernel=gabor_kernel(
                kernel_size=kernel_size,
                wavelength=wavelength,
                orientation=orientation,
                std=std,
                aspect_ratio=aspect_ratio,
                channels=channels,
                off_center=off_center
            ),
                **kwargs
            )