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

class empty_cls:
    pass
class empty_monitor:
    def __init__(self):
        self.variables = []
        self.obj = empty_cls()
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
            if i[-2:]=='3D':
                self.axes[i] = plt.subplot(grid, projection='3d')
            else:
                self.axes[i] = plt.subplot(grid)

    def __getitem__(self, name):
        return self.axes[name]

    def get_monitor(self, monitor=None):
        if monitor is None:
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
        if type(ax) is str:
            ax = self[ax]
        return ax

    def set_labels(self, ax, x_label=None, y_label=None, title=None, x=None, y=None,
                is_3d=False, z_label=None, z=None):
        if self.auto_label:
            if y_label==None:
                y_label = y
            if x_label==None:
                x_label = x
            if z_label==None:
                z_label = z
            if title==None:
                title = f"{y_label}-{x_label}"
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        if is_3d:
            ax.set_zlabel(z_label)

    def get_axis_data(self, z, data=None, monitor=None, repeat_till=None, default=None, **args):
        if data is None:
            data = {}
        monitor = self.get_monitor(monitor)
        output = None
        if z in data:
            output = data[z]
        elif z in monitor.variables:
            output = monitor.get(z).clone()
        else:
            output = default
        if repeat_till!=None:
            if repeat_till<=len(output):
                output = output[:repeat_till]
            else:
                dii = DII(data)
                output = [next(dii)[z] for i in range(repeat_till)]
        return output

    def get_data(self, y, x='time', monitor=None, t0=0, dt=None, repeat_till=None,
                black_line=None, blue_line=None, red_line=None, **args):
        output = {}
        output[y] = self.get_axis_data(z=y, monitor=monitor, default=None, repeat_till=repeat_till, **args)
        dt = self.get_dt(dt, monitor)
        output[x] = self.get_axis_data(z=x, monitor=monitor, repeat_till=repeat_till,
                                    default=np.arange(t0,len(output[y])*dt, dt), **args)
        for line in [black_line, blue_line, red_line]:
            if line!=None:
                output[line] = self.get_axis_data(line, monitor=monitor, repeat_till=len(output[x]), **args)
        return output

    def set_limits(self, ax, x_lim=None, y_lim=None, x=None, y=None, data=None,
                is_3d=False, z_lim=None, z=None):
        if data is None:
            data = {}
        if x_lim=='fit':
            x_lim = [min(data[x]), max(data[x])]
        if y_lim=='fit':
            y_lim = [min(data[y]), max(data[y])]
        if z_lim=='fit' and is_3d:
            z_lim = [min(data[z]), max(data[z])]
        if x_lim is not None:
            ax.set_xlim(x_lim)
        if y_lim is not None:
            ax.set_ylim(y_lim)
        if z_lim is not None and is_3d:
            ax.set_zlim(z_lim)

    def set_axes_visibility(self, ax, x_vis=True, y_vis=True, is_3d=False, z_vis=True):
        if not x_vis:
            ax.get_xaxis().set_ticks([])
        if not y_vis:
            ax.get_yaxis().set_ticks([])
        if is_3d and not z_vis:
            ax.get_zaxis().set_ticks([])

    def plot_extra_lines(self, ax, data, blue_line=None, black_line=None, red_line=None,
                        blue_line_alpha=0.5, black_line_alpha=0.5, red_line_alpha=0.5,
                        blue_line_label=None, black_line_label=None, red_line_label=None):
        if blue_line!=None:
            ax.plot(data[blue_line], 'b--', label=blue_line_label if blue_line_label!=None else blue_line, alpha=blue_line_alpha)
        if black_line!=None:
            ax.plot(data[black_line], 'k-.', label=black_line_label if black_line_label!=None else black_line, alpha=black_line_alpha)
        if red_line!=None:
            ax.plot(data[red_line], 'r:', label=red_line_label if red_line_label!=None else red_line, alpha=red_line_alpha)


    def plot(self, ax, additive=False, plot_type="plot",
            y=None, x='time', data=None, monitor=None, dt:float=None, t0=0,
            title=None, x_label=None, y_label=None, x_vis=True, y_vis=True,
            x_lim=None, y_lim=None, repeat_till=None,
            blue_line=None, black_line=None, red_line=None,
            blue_line_alpha=0.5, black_line_alpha=0.5, red_line_alpha=0.5,
            blue_line_label=None, black_line_label=None, red_line_label=None, **args):
        if data is None:
            data = {}
        monitor = self.get_monitor(monitor)
        dt = self.get_dt(dt, monitor)
        if y==None: y = str(ax)
        data = self.get_data(y=y, x=x, data=data, monitor=monitor, repeat_till=repeat_till, t0=t0, dt=dt,
                            blue_line=blue_line, black_line=black_line, red_line=red_line)
        ax = self.get_ax(ax)
        if plot_type == "plot":
            ax.plot(data[x], data[y], **args)
        elif plot_type == "scatter":
            ax.scatter(data[x], data[y], **args)
        if not additive:
            self.set_labels(ax, x_label, y_label, title, x, y)
            self.set_limits(ax, x_lim, y_lim, x, y, data)
            self.set_axes_visibility(ax, x_vis, y_vis)
            self.plot_extra_lines(ax, data=data, blue_line=blue_line, black_line=black_line, red_line=red_line,
                blue_line_alpha=blue_line_alpha, black_line_alpha=black_line_alpha, red_line_alpha=red_line_alpha,
                blue_line_label=blue_line_label, black_line_label=black_line_label, red_line_label=red_line_label)
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

    def neuron_voltage(self, ax, y='u', monitor=None, y_label='u', x_label='time', threshold='spike_threshold', **args):
        monitor = self.get_monitor(monitor)
        if monitor is not None:
            data = {
                "Resting Potential": [monitor.obj.u_rest.tolist()],
                "Threshold": [getattr(monitor.obj, threshold).tolist()],
            }
        self.plot(ax, y=y, data=data, monitor=monitor, color='green', y_label=y_label, x_label=x_label, x_lim='fit',
                black_line="Resting Potential", blue_line="Threshold", **args)
        ax = self.get_ax(ax)
        ax.legend()

    def neuron_spike(self, ax, y='s', x='time', y_label='spikes', **args):
        data = self.get_data(y=y, x=x, **args)
        x_data = np.array(data[x])
        y_data = np.array(data[y])
        x_data = x_data[y_data.reshape(x_data.shape)]
        self.plot(ax, y=y, x=x, data={x: x_data, y:[1]*x_data.shape[0]}, color='r',
                y_label=y_label, y_vis=False, x_lim=[min(data[x]), max(data[x])],
                plot_type="scatter", **args)

    def adaptation_current_dynamic(self, ax, y='w', monitor=None, additive=False, y_label='w', x_label='time', alpha=.4, **args):
        if additive:
            self.plot(ax, y=y, monitor=monitor, color='red', alpha=alpha, label=y_label, additive=True, **args)
        else:
            self.plot(ax, y=y, monitor=monitor, y_label=y_label, x_label=x_label, x_lim='fit', color='red', **args)

    def population_activity_raster(self, ax, y='s', y_label='spikes', start=0, t0=0, selection=None, s=1, x_lim=None, **args):
        data = self.get_data(y=y, t0=t0, **args)
        y = data[y]
        y = y[:,selection]
        y = y.reshape(y.shape[0], -1)
        if x_lim is None:
            x_lim = (0, y.shape[0])
        x,y = np.where(y)
        x += t0
        y += start
        ax = self.get_ax(ax)
        self.plot(ax, data={'x':x, 'y':y}, x='x', y='y', plot_type="scatter", s=s, y_label=y_label, x_lim=x_lim, **args)

    def population_activity(self, ax, y='s', y_label='activity', selection=None, data=None, alpha=.3, **args):
        if data is None:
            data = {}
        data_y = self.get_data(y=y, data=data, **args)
        y = data_y[y]
        y = y[:,selection]
        y = y.reshape(y.shape[0], -1)
        y = y.sum(axis=1)
        data[y] = y
        self.plot(ax, y=y, data=data, y_label=y_label, alpha=alpha, **args)

    def population_plot(self, ax, y='population', population_alpha=None, color='b', alpha=1, additive=False,
                        aggregation=lambda x: x.mean(axis=1), data=None, **args):
        if data is None:
            data = {}
        data = self.get_data(y=y, data=data, **args)
        if type(data[y])==type([]):
            data[y] = np.array(data[y])
        data['population'] = data[y].reshape(data[y].shape[0],-1)
        if population_alpha is None:
            population_alpha = 1/data['population'].shape[1]
        if aggregation is not None:
            data['vector'] = aggregation(data['population'])
            self.plot(ax, y='vector', additive=additive, data=data, color=color, alpha=alpha, **args)
            self.plot(ax, y='population', additive=True, data=data, color=color, alpha=population_alpha)
        else:
            self.plot(ax, y='population', additive=False, data=data, color=color, alpha=population_alpha, **args)

    def current_dynamic(self, ax, I=None, y='I', y_label='Current', data=None, x_lim='fit', x_label='time', **args):
        if data is None:
            data = {}
        if I is not None:
            data[y] = I
        self.population_plot(ax, y=y, data=data, x_label=x_label, x_lim=x_lim, y_label=y_label, **args)

    def dendrite_current(self, ax, I=None, y='I', y_label='Dendrite Current', data=None, x_lim='fit', x_label='time', **args):
        if data is None:
            data = {}
        if I is not None:
            data[y] = I
        self.population_plot(ax, y=y, data=data, x_label=x_label, x_lim=x_lim, y_label=y_label,
            aggregation=lambda x: x.sum(axis=1), **args)

    def imshow(self, ax, im, aspect='auto', additive=False,
            title='', x_label=None, y_label=None,
            x_vis=True, y_vis=True, x_lim=None, y_lim=None, **args):
        ax = self.get_ax(ax)
        ax.imshow(im, aspect=aspect, **args)
        ax.set_title(title)
        self.set_axes_visibility(ax, x_vis=False, y_vis=False)
        if not additive:
            self.set_labels(ax, x_label, y_label, title)
            self.set_limits(ax, x_lim, y_lim)
            self.set_axes_visibility(ax, x_vis, y_vis)

    def spike_response_function(self, ax, y='e', y_label='Spike Response', x_lim='fit', **args):
        self.population_plot(ax, y=y, y_label=y_label, x_lim=x_lim, **args)

    def get_3d_data(self, z='z', y='y', x='x', monitor=None, **args):
        output = {}
        output[z] = self.get_axis_data(z=z, monitor=monitor, default=None, **args)
        output[y] = self.get_axis_data(z=y, monitor=monitor, default=None, **args)
        output[x] = self.get_axis_data(z=x, monitor=monitor, default=None, **args)
        return output

    def scatter_3d(self, ax, additive=False,
            y='y', x='x', z='z', data=None, monitor=None,
            title=None, x_label=None, y_label=None, z_label=None,
            x_vis=True, y_vis=True, z_vis=True,
            x_r=False, y_r=False, z_r=False,
            x_lim=None, y_lim=None, z_lim=None, **args):
        if data is None:
            data = {}
        monitor = self.get_monitor(monitor)
        if z==None: z = str(ax)
        data = self.get_3d_data(y=y, x=x, z=z, data=data, monitor=monitor)
        ax = self.get_ax(ax)
        ax.scatter(data[x], data[y], data[z], **args)
        if not additive:
            self.set_labels(ax, x_label, y_label, title, x, y, is_3d=True, z_label=z_label, z=z)
            self.set_limits(ax, x_lim, y_lim, x, y, data, is_3d=True, z_lim=z_lim, z=z)
            self.set_axes_visibility(ax, x_vis, y_vis, is_3d=True, z_vis=z_vis)
        axes = ax.axes
        if x_r:
            axes.invert_xaxis()
        if z_r:
            axes.invert_zaxis()
        if y_r:
            axes.invert_yaxis()

    def population_activity_3d_raster(self, ax, z='s', x='x', y='y', y_label='spikes', z_label='time',
            reduction=1, **args):
        data = self.get_3d_data(z='s', x='x', y='y', **args)
        d = data[z]
        d = d.reshape(d.shape[0],-1)
        xe,ye = np.where(d)
        ye = np.array(ye)
        ye = ye//reduction
        xd = (np.array(ye)//data[z].shape[0])
        yd = (np.array(ye)%data[z].shape[1])
        zd = np.array(xe)
        self.scatter_3d(ax, data={z:zd, x:xd, y:yd}, z=z, x=x, y=y, z_label=z_label, **args)

    def surface_3d(self, ax, z='z', x='x', y='y', data=None, monitor=None, additive=False,
            title=None, x_label=None, y_label=None, z_label=None,
            x_vis=True, y_vis=True, z_vis=True,
            x_r=False, y_r=False, z_r=False,
            x_lim=None, y_lim=None, z_lim=None, **args):
        if data is None:
            data = {}
        monitor = self.get_monitor(monitor)
        if z==None: z = str(ax)
        data = self.get_3d_data(z=z, y=y, x=x, data=data, monitor=monitor)
        Z = data[z].numpy()
        X = np.arange(Z.shape[0])
        Y = np.arange(Z.shape[1])
        X, Y = np.meshgrid(X, Y)
        ax = self.get_ax(ax)
        surf = ax.plot_surface(X, Y, Z, **args)
        if not additive:
            self.set_labels(ax, x_label, y_label, title, x, y, is_3d=True, z_label=z_label, z=z)
            self.set_limits(ax, x_lim, y_lim, x, y, data, is_3d=True, z_lim=z_lim, z=z)
            self.set_axes_visibility(ax, x_vis, y_vis, is_3d=True, z_vis=z_vis)
        axes = ax.axes
        if x_r:
            axes.invert_xaxis()
        if z_r:
            axes.invert_zaxis()
        if y_r:
            axes.invert_yaxis()
        return surf