<div align="center">
  <img src="https://github.com/BehzadShayegh/Spiral/blob/main/docs/logos/spiral_purple.png"/>
</div>

***

A python package for spiking neural network simulation using PyTorch on CUDA or CPU. This package tries to bring its design as close as possible to biological observations of how the nervous system functions. We have a long way to go for a mature package that covers all needs you might have dealing with spiking neural networks. Yet this package is still an infant child. Some of the valuable simulations that this design claims can easily support, unlike other available tools, are:
* use of spiking and classical neurons in the same network,
* simulating action potential response functions and
* training in time latency (where implementation is ready but still empty).

By breaking components into several interconnected modules, the package has been able to
* bring its design closer to biological documentation, 
* increases network creation flexibility and 
* easily creates real-world features such as the delayed transmission of neuromodulators (delaying axons can be connected to neuromodulatory neurons).

This design brings a great deal of re-usability so that structures that you define as encoders or interneurons can be used as neuromodulatory neurons. It is also possible to define one response function and use it at axon terminals to simulate the release process of neurotransmitters or at dendritic spins as synaptic taggers.

This package still has a minimal core and needs to be developed. But it has put much effort into simplifying the modeling process. Providing a suitable object-oriented design and creating efficient infrastructures has dramatically simplified implementing different models and combining them with other ones.

We are very welcome to receive assistance in the development of this toolkit. Also, if the features briefly mentioned are helpful to you but the development of this tool is still inefficient, we invite you to wait for the following versions.

* **Note:** This repository has a complicated history that makes it so heavy. We will prune its history shortly, but by then, we recommend you clone it using `git clone --bare` to avoid cloning useless history.
