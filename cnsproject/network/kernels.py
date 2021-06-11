import torch
from typing import Union, Iterable


def gaussian_kernel(
        kernel_size: int = 3,
        std: Union[float,torch.Tensor] = 1.,
        channels: int = None,
    ) -> torch.Tensor:
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    mean = (kernel_size - 1)/2.
    variance = std**2.
    pi = torch.acos(torch.zeros(1)).item() * 2
    kernel = (1./(2.*pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2*variance)
                      )
    kernel = kernel / torch.sum(kernel)
    if channels is not None:
        kernel = kernel.view(1, *kernel.shape) #channels - x - y
        kernel = kernel.repeat(channels, 1, 1)
    return kernel


def DoG_kernel(
        kernel_size: int = 3,
        std1: Union[float,torch.Tensor] = 1.,
        std2: Union[float,torch.Tensor] = 2.,
        channels: int = None,
        off_center=False,
    ) -> torch.Tensor:
    kernel = gaussian_kernel(kernel_size=kernel_size, std=std1, channels=channels) \
            - gaussian_kernel(kernel_size=kernel_size, std=std2, channels=channels)
    kernel -= kernel.mean()
    if off_center:
        kernel *= -1
    return kernel


def gabor_kernel(
        kernel_size: int = 3,
        wavelength: Union[float,torch.Tensor] = 1.,
        orientation: torch.Tensor = torch.tensor(0.),
        std: Union[float,torch.Tensor] = 1.,
        aspect_ratio: Union[float,torch.Tensor] = 1.,
        channels: int = None,
        off_center=False,
    ) -> torch.Tensor:
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    x_grid,y_grid = x_grid.float(),y_grid.float()
    x_grid -= x_grid.mean()
    y_grid -= y_grid.mean()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    mean = (kernel_size - 1)/2.
    variance = std**2.
    pi = torch.acos(torch.zeros(1)).item() * 2
    x = xy_grid.T[0].T
    y = xy_grid.T[1].T
    X = x*torch.cos(orientation) + y*torch.sin(orientation)
    Y = -x*torch.sin(orientation) + y*torch.cos(orientation)
    kernel = torch.exp(
                -(X**2 + (aspect_ratio**2) * (Y**2)) /\
                (2*variance)
            ) * torch.cos(2*pi*X/wavelength)
    kernel = kernel / torch.sum(kernel)
    if channels is not None:
        kernel = kernel.view(1, *kernel.shape) #channels - x - y
        kernel = kernel.repeat(channels, 1, 1)
    kernel -= kernel.mean()
    if off_center:
        kernel *= -1
    return kernel