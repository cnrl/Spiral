from abc import abstractmethod
from typing import Union, Iterable, Callable
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




class CoreCentricFilter(AbstractFilter):
    def __init__(
        self,
        core: Callable, # input: tensor of shape ([batch,channel,] ...) that ... is `dims` dimentional and batch exists if batch_recipient_core and channel exists if channel_recipient_core
        dims: int = 2,
        batch_inputing: bool = False,
        batch_recipient_core: bool = True, #if not batch_inputing, it will run unsqueeze_(0) on inputs befor sending them to core
        batch_outputing: bool = False, #if False and batch_recipient_core, it will drop first dim of core output before redirecting it to object output
        channel_inputing: bool = False,
        channel_recipient_core: bool = True, #if not channel_inputing, it will run unsqueeze_(0) on inputs befor checking batchs
        channel_outputing: bool = False, #if False and channel_recipient_core, it will drop second dim of core output before redirecting it to object output
        padding: Iterable = (),
        post_reshape_transform: Callable = lambda x: x,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.post_reshape_transform = post_reshape_transform
        self.dims = dims
        assert not (batch_inputing and not batch_recipient_core), "What should I do with batch dimention?"
        assert not (not batch_recipient_core and batch_outputing), "Should I generate one dimention?"
        self.batch_inputing = batch_inputing
        self.batch_recipient_core = batch_recipient_core
        self.batch_outputing = batch_outputing
        assert not (channel_inputing and not channel_recipient_core), "What should I do with batch dimention?"
        assert not (not channel_recipient_core and channel_outputing), "Should I generate one dimention?"
        self.channel_inputing = channel_inputing
        self.channel_recipient_core = channel_recipient_core
        self.channel_outputing = channel_outputing
        self.core = core
        self.padding = padding
        self.add_module('core', self.core)


    def input_reshape(self, tensor: torch.Tensor) -> torch.Tensor: #([batch,channel,] ...)
        assert len(tensor.shape)==self.dims+self.channel_inputing+self.batch_inputing, "tensor shape is wrong: it should be ([batch,channel,] ...)"
        if self.channel_recipient_core and not self.channel_inputing:
            tensor = tensor.reshape(1, *tensor.shape)
        if self.batch_recipient_core and not self.batch_inputing:
            tensor = tensor.reshape(1, *tensor.shape)
        return tensor


    def output_reshape(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.batch_recipient_core:
            if self.channel_recipient_core and not self.channel_outputing:
                return tensor[0]
            else:
                return tensor
        if self.batch_recipient_core and not self.batch_outputing:
            tensor = tensor[0]
            if self.channel_recipient_core and not self.channel_outputing:
                return tensor[0]
            else:
                return tensor
        else:
            if self.channel_recipient_core and not self.channel_outputing:
                return tensor[:,0]
            else:
                return tensor


    def pad(self, tensor: torch.Tensor) -> torch.Tensor:
        for i,pad in enumerate(self.padding):
            shape = tensor.shape
            shape[i] = pad
            tensor = torch.cat([torch.zeros(shape),tensor,torch.zeros(shape)], axis=0)
        return tensor

    def __call__(self, data: torch.Tensor) -> torch.Tensor: # data in shape=(b,c,x,y) as (batch,channels,height,width)
        data = super().__call__(data)
        data = self.input_reshape(data)
        data = self.post_reshape_transform(data)
        data = self.core(data).detach()
        return self.output_reshape(data)




class Conv2DFilter(CoreCentricFilter):
    def __init__(
        self,
        kernel, #([batch,channel,]size,size)
        transform: Callable = lambda x: x,
        post_reshape_transform: Callable = lambda x: x,
        batch_inputing: bool = False,
        batch_outputing: bool = False,
        channel_inputing: bool = None,
        channel_outputing: bool = None,
        **kwargs
    ) -> None:
        super().__init__(
            dims=2,
            batch_inputing=batch_inputing,
            batch_recipient_core=True,
            batch_outputing=batch_outputing,
            channel_inputing=channel_inputing if channel_inputing is not None else len(kernel.shape)>=3,
            channel_recipient_core=True,
            channel_outputing=channel_outputing if channel_outputing is not None else len(kernel.shape)>=4,
            transform=transform,
            post_reshape_transform=post_reshape_transform,
            core=Conv2DFilter.conv_init(kernel=kernel, **kwargs),
        )


    def conv_init(kernel, **kwargs) -> torch.nn.Conv2d:
        while len(kernel.shape)<4:
            kernel = kernel.unsqueeze_(0)
        filt = torch.nn.Conv2d(
            in_channels=kernel.shape[1],
            out_channels=kernel.shape[0],
            kernel_size=(),
            **kwargs
        )
        filt.weight.data = kernel
        filt.weight.requires_grad = False
        return filt