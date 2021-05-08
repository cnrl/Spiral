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
        self.state_variables = []
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

    def get_data(self, y, x='time', data={}, monitor=None, repeat_till=None, t0=0, dt=None,
                black_line=None, blue_line=None, red_line=None, **args):
        output = {}
        output[y] = self.get_axis_data(z=y, data=data, monitor=monitor, repeat_till=repeat_till, default=None)
        dt = self.get_dt(dt, monitor)
        output[x] = self.get_axis_data(z=x, data=data, monitor=monitor, repeat_till=repeat_till,
                                    default=np.arange(t0,len(output[y])*dt, dt))
        for line in [black_line, blue_line, red_line]:
            if line!=None:
                output[line] = self.get_axis_data(line, data, monitor, repeat_till=len(output[x]))
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

    def plot_extra_lines(self, ax, data, blue_line=None, black_line=None, red_line=None,
                        blue_line_alpha=0.5, black_line_alpha=0.5, red_line_alpha=0.5,
                        blue_line_label=None, black_line_label=None, red_line_label=None):
        if blue_line!=None:
            ax.plot(data[blue_line], 'b--', label=blue_line_label if blue_line_label!=None else blue_line, alpha=blue_line_alpha)
        if black_line!=None:
            ax.plot(data[black_line], 'k-.', label=black_line_label if black_line_label!=None else black_line, alpha=black_line_alpha)
        if red_line!=None:
            ax.plot(data[red_line], 'r:', label=red_line_label if red_line_label!=None else red_line, alpha=red_line_alpha)


    def plot(self, ax, additive=False,
            y=None, x='time', data={}, monitor=None, dt:float=None, t0=0,
            title=None, x_label=None, y_label=None, x_vis=True, y_vis=True,
            x_lim=None, y_lim=None, repeat_till=None, style='b',
            blue_line=None, black_line=None, red_line=None,
            blue_line_alpha=0.5, black_line_alpha=0.5, red_line_alpha=0.5,
            blue_line_label=None, black_line_label=None, red_line_label=None, **args):

        monitor = self.get_monitor(monitor)
        dt = self.get_dt(dt, monitor)
        if y==None: y = str(ax)
        data = self.get_data(y=y, x=x, data=data, monitor=monitor, repeat_till=repeat_till, t0=t0, dt=dt,
                            blue_line=blue_line, black_line=black_line, red_line=red_line)
        ax = self.get_ax(ax)
        ax.plot(data[x], data[y], style, **args)
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
        data = {
            "Resting Potential": [monitor.obj.u_rest.tolist()],
            "Threshold": [getattr(monitor.obj, threshold).tolist()],
        }
        self.plot(ax, y=y, data=data, monitor=monitor, color='green', y_label=y_label, x_label=x_label, x_lim='fit',
                black_line="Resting Potential", blue_line="Threshold", **args)
        ax = self.get_ax(ax)
        ax.legend()

    def neuron_spike(self, ax, y='s', x='time', **args):
        data = self.get_data(y=y, x=x, **args)
        x_data = np.array(data[x])
        x_data = x_data[data[y]]
        self.plot(ax, y=y, x=x, data={x: x_data, y:[1]*x_data.shape[0]}, style='ro', y_vis=False, x_lim=[min(data[x]), max(data[x])], **args)

    def adaptation_current_dynamic(self, ax, y='w', monitor=None, additive=False, y_label='w', x_label='time', alpha=.4, **args):
        if additive:
            self.plot(ax, y=y, monitor=monitor, color='red', alpha=alpha, label=y_label, additive=True, **args)
        else:
            self.plot(ax, y=y, monitor=monitor, y_label=y_label, x_label=x_label, x_lim='fit', color='red', **args)

    def get_excitatory_flag(self, name='is_excitatory', data={}, y_shape=None, monitor=None):
        if name in data:
            return data[name]
        monitor = self.get_monitor(monitor)
        if name in monitor.obj.__dict__['_buffers']:
            return monitor.obj.__dict__['_buffers'][name]
        return np.ones(y.shape, dtype=bool)

    def get_population_activity_raster_data(self, y='s', x='time', excitatory_flag='is_excitatory', start=0, monitor=None, **args):
        data = self.get_data(y=y, x=x, monitor=monitor, **args)
        excitatory_flag = self.get_excitatory_flag(name=excitatory_flag, y_shape=data[y][0].shape, monitor=monitor)
        ce = data[y][0][excitatory_flag].numel()
        ci = data[y][0][~excitatory_flag].numel()
        xe,ye,xi,yi = [],[],[],[]
        for index,t in enumerate(data[x]):
            yt = data[y][index]
            e = yt[excitatory_flag].reshape(-1)
            i = yt[~excitatory_flag].reshape(-1)
            se = np.where(e)[0] + start
            si = np.where(i)[0] + ce + start
            ye += se.tolist()
            yi += si.tolist()
            xe += (np.ones(se.shape)*t).tolist()
            xi += (np.ones(si.shape)*t).tolist()
        return xe,ye,ce,xi,yi,ci

    def population_activity_raster(self, ax, y='s', x='time', label_prefix='', start=0,
            data={}, monitor=None, repeat_till=None, t0=0, dt=None, color={}, marker={},
            title=None, x_label=None, y_label=None, x_vis=True, y_vis=False,
            x_lim=None, y_lim=None, s=1, additive=False, excitatory_flag='is_excitatory', **args):
        if type(color)==type(''):
            color = {'e': color, 'i': color}
        if type(marker)==type(''):
            marker = {'e': marker, 'i': marker}

        xe,ye,ce,xi,yi,ci = self.get_population_activity_raster_data(y=y, x=x, start=start,
            data=data, monitor=monitor, repeat_till=repeat_till, t0=t0, dt=dt, excitatory_flag=excitatory_flag)
        ax = self.get_ax(ax)
        if len(xe)>0:
            ax.scatter(xe, ye, color=color.get('e','g'), marker=marker.get('e','o'), s=s, label=label_prefix+'excitatory', **args)
        if len(xi)>0:
            ax.scatter(xi, yi, color=color.get('i','r'), marker=marker.get('e','o'), s=s, label=label_prefix+'inhibitory', **args)
        if not additive:
            self.set_labels(ax, x_label, y_label, title, x, y)
            self.set_limits(ax, x_lim, y_lim, x, y, data)
            self.set_axes_visibility(ax, x_vis, y_vis)
        return xe,ye,ce,xi,yi,ci

    def population_activity(self, ax, raster_data=None, y='a', x='time', y_label='s count',
                ei_split=False, label='', color='b', ei_color={},
                data={}, monitor=None, repeat_till=None, t0=0, dt=None, alpha=.3, additive=False, **args):
        if raster_data is None:
            raster_data = self.get_population_activity_raster_data(y=y, x=x, 
                data=data, monitor=monitor, repeat_till=repeat_till, t0=t0, dt=dt)
        xe,_,ce,xi,_,ci = raster_data

        ie = np.unique(xe+xi, return_counts=True)
        iex,iey = ie
        if iex.shape[0]>0:
            ie = np.zeros(int(1+iex.max()-iex.min()))
            ie[(iex-iex.min()).astype(int)] = iey
            iex = np.arange(iex.min(),iex.max()+1)

            range_ = (int(iex.min()), int(iex.max())+1)
            self.plot(ax, additive=additive, y=y, x=x, data={x: iex, y: ie/(ce+ci)}, y_label=y_label, color=color, label=label, alpha=alpha, **args)

        if ei_split:
            e = np.unique(xe, return_counts=True)
            ex,ey = e
            if ex.shape[0]>0:
                e = np.zeros(int(1+ex.max()-ex.min()))
                e[(ex-ex.min()).astype(int)] = ey
                ex = np.arange(ex.min(),ex.max()+1)
                self.plot(ax, additive=True, y=y, x=x, data={x: ex, y: e/ce}, color=ei_color.get('e','g'), label=label+'excitatory', alpha=alpha, **args)

            i = np.unique(xi, return_counts=True)
            ix,iy = i
            if ix.shape[0]>0:
                i = np.zeros(int(1+ix.max()-ix.min()))
                i[(ix-ix.min()).astype(int)] = iy
                ix = np.arange(ix.min(),ix.max()+1)
                self.plot(ax, additive=True, y=y, x=x, data={x: ix, y: i/ci}, color=ei_color.get('i','r'), label=label+'inhibitory', alpha=alpha, **args)

            ax = self.get_ax(ax)
            ax.legend()
            return iex,ie,ce+ci,ex,e,ce,ix,i,ci
        return iex,ie,ce+ci

    def population_plot(self, ax, vector, data={}, population_alpha=0.01, color='b', alpha=1, additive=False, **args):
        if type(vector)==type([]):
            vector = np.array(vector)
        data['population'] = vector.reshape(vector.shape[0],-1)
        data['vector'] = data['population'].mean(axis=1)
        self.plot(ax, y='vector', additive=additive, data=data, color=color, alpha=alpha, **args)
        self.plot(ax, y='population', additive=True, data=data, color=color, alpha=population_alpha)

    def current_dynamic(self, ax, I=None, y='I', y_label='Current', data={}, x_label='time', **args):
        if type(I)==type(None):
            I = data[y]
        self.population_plot(ax, vector=I, data=data, x_label=x_label, x_lim='fit', y_label=y_label, **args)

    def imshow(self, ax, im, title='', aspect='auto', **args):
        ax = self.get_ax(ax)
        ax.imshow(im, aspect=aspect, **args)
        ax.set_title(title)
        self.set_axes_visibility(ax, x_vis=False, y_vis=False)