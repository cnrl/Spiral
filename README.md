<div align="center">
  <img src="https://github.com/BehzadShayegh/Spiral/blob/main/docs/logos/spiral_purple.png"/>
</div>

***

A python package for spiking neural network simulation using [PyTorch](http://pytorch.org/) on cuda or CPU.
This package tries to bring its design as close as possible to biological observations of the functioning of the nervous system.
We have a long way to go and this package is still very toddler.
Valuable simulations that this design claims can easily support, unlike other available tools, are "use of spiking and classical neurons in samenetwork", "simulating action potential response functions", and "training in time latency (where implementation is ready but still empty)".

By breaking components into several interconnected modules, the package has been able to bring its design closer to biological documentations, increases network creation flexibility, and easily creates real-world features such as delayed transmission of neuromodulators (by connecting delaying axons to neuromodulatory neurons).
This design creates a great deal of reusability so that structures that are defined as encoders or interneurons can simply be used as neuromodulatory neurons.
This design provides so much reusability that structures defined as encoders or between neurons can simply be used as neural modulating neurons. It is also possible to define one response function and use it at axon terminals to simulate the release process of neurotransmitters or at dendritic spins as synaptic taggers.

This package still has a very small core and needs to be developed. But it has put a lot of effort into simplifying the modeling process. By providing a suitable object-oriented design and creating efficient infrastructures, it has greatly simplified the process of implementing different models and combining them with other models.

We are very welcome to receive assistance in the development of this tool. Also, if the features briefly mentioned are useful to you but the development of this tool is still not enough for you, we invite you to wait for the growth of this tool.

## Requirements

- Python 3.6
- `requirements.txt` (Not updated)
