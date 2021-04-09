"""
Module for visualization and plotting.

3. Raster plot of spikes in a neural population.
4. Convolutional weight demonstration.
5. Weight change through time.
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from ..utils import DII

class empty_monitor:
    def __init__(self):
        self.state_variables = []
    def get_time_info(self, time=None, dt=None):
        return None,None

class Plotter:
    def __init__(self, positions:list, monitor=empty_monitor(), dt:float=None, auto_label=False, **args):
        self.monitor = monitor
        self.dt = dt
        self.auto_label = auto_label
        self.make_axes(positions, **args)

    def make_axes(self, positions, **args):
        positions = np.array(positions)
        if len(positions.shape) == 1:
            positions = positions.reshape(1,-1)
        positions[positions == None] = 'None'
        self.axes = dict()
        gs = gridspec.GridSpec(*positions.shape, **args)
        for i in np.unique(positions):
            if i == 'None':
                continue
            y,x = np.where(positions == i)
            y_min = y.min()
            y_max = y.max()
            x_min = x.min()
            x_max = x.max()
            grid = gs[y_min:y_max+1, x_min:x_max+1]
            self.axes[i] = plt.subplot(grid)

    def __getitem__(self, name):
        return self.axes[name]

    def get_monitor(self, monitor=None):
        if type(monitor)==type(None):
            monitor = self.monitor
        return monitor

    def get_dt(self, dt=None, monitor=None):
        if dt==None:
            dt = self.dt
        if dt==None:
            monitor = self.get_monitor(monitor)
            dt = monitor.get_time_info(dt=dt)[1]
        if dt==None:
            dt=1
        return dt
    
    def get_ax(self, ax):
        if type(ax)!=type(next(iter(self.axes.values()))):
            ax = self[ax]
        return ax

    def set_labels(self, ax, x_label=None, y_label=None, title=None, x=None, y=None):
        if self.auto_label:
            if y_label==None:
                y_label = y
            if x_label==None:
                x_label = x
            if title==None:
                title = f"{y_label}-{x_label}"
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)

    def get_axis_data(self, z, data={}, monitor=None, repeat_till=None, default=None):
        monitor = self.get_monitor(monitor)
        output = None
        if z in data:
            output = data[z]
        elif z in monitor.state_variables:
            output = monitor.get(z)
        else:
            output = default
        if repeat_till!=None:
            if repeat_till<=len(output):
                output = output[:repeat_till]
            else:
                dii = DII(data)
                output = [next(dii)[z] for i in range(repeat_till)]
        return output

    def get_data(self, y, x='time', data={}, monitor=None, repeat_till=None, t0=0, dt=None, **args):
        output = {}
        output[y] = self.get_axis_data(z=y, data=data, monitor=monitor, repeat_till=repeat_till, default=None)
        dt = self.get_dt(dt, monitor)
        output[x] = self.get_axis_data(z=x, data=data, monitor=monitor, repeat_till=repeat_till,
                                    default=np.arange(t0,len(output[y])*dt, dt))
        return output

    def set_limits(self, ax, x_lim=None, y_lim=None, x=None, y=None, data={}):
        if x_lim=='fit':
            x_lim = [min(data[x]), max(data[x])]
        if y_lim=='fit':
            y_lim = [min(data[y]), max(data[y])]
        if type(x_lim)!=type(None):
            ax.set_xlim(x_lim)
        if type(y_lim)!=type(None):
            ax.set_ylim(y_lim)

    def set_axes_visibility(self, ax, x_vis=True, y_vis=True):
        if not x_vis:
            ax.get_xaxis().set_ticks([])
        if not y_vis:
            ax.get_yaxis().set_ticks([])


    def plot(self, ax,
            y=None, x='time', data={}, monitor=None, dt:float=None, t0=0,
            title=None, x_label=None, y_label=None, x_vis=True, y_vis=True,
            x_lim=None, y_lim=None, repeat_till=None, style='b', **args):

        monitor = self.get_monitor(monitor)
        dt = self.get_dt(dt, monitor)
        if y==None: y = str(ax)
        data = self.get_data(y=y, x=x, data=data, monitor=monitor, repeat_till=repeat_till, t0=t0, dt=dt)
        ax = self.get_ax(ax)
        ax.plot(data[x], data[y], style, **args)
        self.set_labels(ax, x_label, y_label, title, x, y)
        self.set_limits(ax, x_lim, y_lim, x, y, data)
        self.set_axes_visibility(ax, x_vis, y_vis)
        return len(data[x])

    def show(self):
        plt.show()

    def F_I_curve(self, ax, data: dict, **args):
        self.plot(ax,
                y='f',
                x='I',
                data={'f': data.values(), 'I': data.keys()},
                title='frequency-current relation',
                y_label='spike frequency (1/s)',
                x_label='I (mV)',
                color='red',
                **args)

    def neuron_voltage(self, ax, y='u', monitor=None, spike_threshold='spike_threshold', resting_potential='u_rest',
                        y_label='u', x_label='time', **args):
        monitor = self.get_monitor(monitor)
        length = self.plot(ax, y=y, monitor=monitor, color='green', y_label=y_label, x_label=x_label, x_lim='fit', **args)
        ax = self.get_ax(ax)
        ax.plot([getattr(monitor.obj, spike_threshold)]*length, 'b--')
        ax.plot([getattr(monitor.obj, resting_potential)]*length, 'k-.')

    def neuron_spike(self, ax, y='s', x='time', **args):
        data = self.get_data(y=y, x=x, **args)
        x_data = np.array(data[x])
        x_data = x_data[data[y]]
        self.plot(ax, y=y, x=x, data={x: x_data, y:[1]*x_data.shape[0]}, style='ro', y_vis=False, x_lim=[min(data[x]), max(data[x])], **args)

    def current_dynamic(self, ax, I=None, y='I', y_label='I', data={}, x_label='time', **args):
        if type(I)!=type(None):
            data[y] = I
        self.plot(ax, y=y, data=data, x_label=x_label, x_lim='fit', y_label=y_label, **args)
