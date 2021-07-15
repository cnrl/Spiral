"""
"""
import torch


class Plotter:
    def __init__(
        self,
        ax
    ):
        self.ax = ax


    def label(
        self,
        x=None,
        y=None,
        z=None,
        title=None
    ):
        if x is not None:
            self.ax.set_xlabel(x)
        if y is not None:
            self.ax.set_ylabel(y)
        if z is not None:
            self.ax.set_zlabel(z)
        if title is not None:
            self.ax.set_title(title)
        return self.ax


    def limit(
        self,
        x=None,
        y=None,
        z=None
    ):
        if x is not None:
            self.ax.set_xlim(x)
        if y is not None:
            self.ax.set_ylim(y)
        if z is not None:
            self.ax.set_zlim(z)
        return self.ax


    def remove_ticks(
        self,
        x=False,
        y=False,
        z=False
    ):
        if x:
            self.ax.get_xaxis().set_ticks([])
        if y:
            self.ax.get_yaxis().set_ticks([])
        if z:
            self.ax.get_zaxis().set_ticks([])
        return self.ax


    from .. import IntegrateAndFireSoma

    def integrate_and_fire_neuron_potential(
        self,
        potential, torch.Tensor,
        neuron: IntegrateAndFireSoma,
    ):
        x = [i*neuron.dt for i in range(len(potential))]
        y = potential
        self.ax.plot(x, y, color='green', label='potential')
        self.ax.plot(neuron.resting_potential, 'k-.', label='resting potential')
        self.ax.plot(neuron.firing_threshold, 'b--', label='firing threshold')
        self.label(y='voltage(mV)', x='time(ms)')
        self.limit(x=(min(x), max(x)))
        ax.legend()
        return self.ax
        


    # def plot_extra_lines(self, ax, data, blue_line=None, black_line=None, red_line=None,
    #                     blue_line_alpha=0.5, black_line_alpha=0.5, red_line_alpha=0.5,
    #                     blue_line_label=None, black_line_label=None, red_line_label=None):
    #     if blue_line!=None:
    #         ax.plot(data[blue_line], 'b--', label=blue_line_label if blue_line_label!=None else blue_line, alpha=blue_line_alpha)
    #     if black_line!=None:
    #         ax.plot(data[black_line], 'k-.', label=black_line_label if black_line_label!=None else black_line, alpha=black_line_alpha)
    #     if red_line!=None:
    #         ax.plot(data[red_line], 'r:', label=red_line_label if red_line_label!=None else red_line, alpha=red_line_alpha)


    # def plot(self, ax, additive=False, plot_type="plot",
    #         y=None, x='time', data=None, monitor=None, dt:float=None, t0=0,
    #         title=None, x_label=None, y_label=None, x_vis=True, y_vis=True,
    #         x_lim=None, y_lim=None, repeat_till=None,
    #         blue_line=None, black_line=None, red_line=None,
    #         blue_line_alpha=0.5, black_line_alpha=0.5, red_line_alpha=0.5,
    #         blue_line_label=None, black_line_label=None, red_line_label=None, **kwargs):
    #     if data is None:
    #         data = {}
    #     monitor = self.get_monitor(monitor)
    #     dt = self.get_dt(dt, monitor)
    #     if y==None: y = str(ax)
    #     data = self.get_data(y=y, x=x, data=data, monitor=monitor, repeat_till=repeat_till, t0=t0, dt=dt,
    #                         blue_line=blue_line, black_line=black_line, red_line=red_line)
    #     ax = self.get_ax(ax)
    #     if plot_type == "plot":
    #         ax.plot(data[x], data[y], **kwargs)
    #     elif plot_type == "scatter":
    #         ax.scatter(data[x], data[y], **kwargs)
    #     if not additive:
    #         self.set_labels(ax, x_label, y_label, title, x, y)
    #         self.set_limits(ax, x_lim, y_lim, x, y, data)
    #         self.set_axes_visibility(ax, x_vis, y_vis)
    #         self.plot_extra_lines(ax, data=data, blue_line=blue_line, black_line=black_line, red_line=red_line,
    #             blue_line_alpha=blue_line_alpha, black_line_alpha=black_line_alpha, red_line_alpha=red_line_alpha,
    #             blue_line_label=blue_line_label, black_line_label=black_line_label, red_line_label=red_line_label)
    #     return len(data[x])

    # def F_I_curve(self, ax, data: dict, **kwargs):
    #     self.plot(ax,
    #             y='f',
    #             x='I',
    #             data={'f': data.values(), 'I': data.keys()},
    #             title='frequency-current relation',
    #             y_label='spike frequency (1/s)',
    #             x_label='I (mV)',
    #             color='red',
    #             **kwargs)



    # def neuron_spike(self, ax, y='s', x='time', y_label='spikes', **kwargs):
    #     data = self.get_data(y=y, x=x, **kwargs)
    #     x_data = np.array(data[x])
    #     y_data = np.array(data[y])
    #     x_data = x_data[y_data.reshape(x_data.shape)]
    #     self.plot(ax, y=y, x=x, data={x: x_data, y:[1]*x_data.shape[0]}, color='r',
    #             y_label=y_label, y_vis=False, x_lim=[min(data[x]), max(data[x])],
    #             plot_type="scatter", **kwargs)

    # def adaptation_current_dynamic(self, ax, y='w', monitor=None, additive=False, y_label='w', x_label='time', alpha=.4, **kwargs):
    #     if additive:
    #         self.plot(ax, y=y, monitor=monitor, color='red', alpha=alpha, label=y_label, additive=True, **kwargs)
    #     else:
    #         self.plot(ax, y=y, monitor=monitor, y_label=y_label, x_label=x_label, x_lim='fit', color='red', **kwargs)

    # def population_activity_raster(self, ax, y='s', y_label='spikes', start=0, t0=0, selection=None, s=1, x_lim=None, **kwargs):
    #     data = self.get_data(y=y, t0=t0, **kwargs)
    #     y = data[y]
    #     y = y[:,selection]
    #     y = y.reshape(y.shape[0], -1)
    #     if x_lim is None:
    #         x_lim = (0, y.shape[0])
    #     x,y = np.where(y)
    #     x += t0
    #     y += start
    #     ax = self.get_ax(ax)
    #     self.plot(ax, data={'x':x, 'y':y}, x='x', y='y', plot_type="scatter", s=s, y_label=y_label, x_lim=x_lim, **kwargs)

    # def population_activity(self, ax, y='s', y_label='activity', selection=None, data=None, alpha=.3, **kwargs):
    #     if data is None:
    #         data = {}
    #     data_y = self.get_data(y=y, data=data, **kwargs)
    #     y = data_y[y]
    #     y = y[:,selection]
    #     y = y.reshape(y.shape[0], -1)
    #     y = y.sum(axis=1)
    #     data[y] = y
    #     self.plot(ax, y=y, data=data, y_label=y_label, alpha=alpha, **kwargs)

    # def population_plot(self, ax, y='population', population_alpha=None, color='b', alpha=1, additive=False,
    #                     aggregation=lambda x: x.mean(axis=1), data=None, **kwargs):
    #     if data is None:
    #         data = {}
    #     data = self.get_data(y=y, data=data, **kwargs)
    #     if type(data[y])==type([]):
    #         data[y] = np.array(data[y])
    #     data['population'] = data[y].reshape(data[y].shape[0],-1)
    #     if population_alpha is None:
    #         population_alpha = 1/data['population'].shape[1]
    #     if aggregation is not None:
    #         data['vector'] = aggregation(data['population'])
    #         self.plot(ax, y='vector', additive=additive, data=data, color=color, alpha=alpha, **kwargs)
    #         self.plot(ax, y='population', additive=True, data=data, color=color, alpha=population_alpha)
    #     else:
    #         self.plot(ax, y='population', additive=False, data=data, color=color, alpha=population_alpha, **kwargs)

    # def current_dynamic(self, ax, I=None, y='I', y_label='Current', data=None, x_lim='fit', x_label='time', **kwargs):
    #     if data is None:
    #         data = {}
    #     if I is not None:
    #         data[y] = I
    #     self.population_plot(ax, y=y, data=data, x_label=x_label, x_lim=x_lim, y_label=y_label, **kwargs)

    # def dendrite_current(self, ax, I=None, y='I', y_label='Dendrite Current', data=None, x_lim='fit', x_label='time', **kwargs):
    #     if data is None:
    #         data = {}
    #     if I is not None:
    #         data[y] = I
    #     self.population_plot(ax, y=y, data=data, x_label=x_label, x_lim=x_lim, y_label=y_label,
    #         aggregation=lambda x: x.sum(axis=1), **kwargs)

    # def imshow(self, ax, im, aspect='auto', additive=False,
    #         title='', x_label=None, y_label=None,
    #         x_vis=True, y_vis=True, x_lim=None, y_lim=None, **kwargs):
    #     ax = self.get_ax(ax)
    #     ax.imshow(im, aspect=aspect, **kwargs)
    #     ax.set_title(title)
    #     self.set_axes_visibility(ax, x_vis=False, y_vis=False)
    #     if not additive:
    #         self.set_labels(ax, x_label, y_label, title)
    #         self.set_limits(ax, x_lim, y_lim)
    #         self.set_axes_visibility(ax, x_vis, y_vis)

    # def spike_response_function(self, ax, y='e', y_label='Spike Response', x_lim='fit', **kwargs):
    #     self.population_plot(ax, y=y, y_label=y_label, x_lim=x_lim, **kwargs)

    # def get_3d_data(self, z='z', y='y', x='x', monitor=None, **kwargs):
    #     output = {}
    #     output[z] = self.get_axis_data(z=z, monitor=monitor, default=None, **kwargs)
    #     output[y] = self.get_axis_data(z=y, monitor=monitor, default=None, **kwargs)
    #     output[x] = self.get_axis_data(z=x, monitor=monitor, default=None, **kwargs)
    #     return output

    # def scatter_3d(self, ax, additive=False,
    #         y='y', x='x', z='z', data=None, monitor=None,
    #         title=None, x_label=None, y_label=None, z_label=None,
    #         x_vis=True, y_vis=True, z_vis=True,
    #         x_r=False, y_r=False, z_r=False,
    #         x_lim=None, y_lim=None, z_lim=None, **kwargs):
    #     if data is None:
    #         data = {}
    #     monitor = self.get_monitor(monitor)
    #     if z==None: z = str(ax)
    #     data = self.get_3d_data(y=y, x=x, z=z, data=data, monitor=monitor)
    #     ax = self.get_ax(ax)
    #     ax.scatter(data[x], data[y], data[z], **kwargs)
    #     if not additive:
    #         self.set_labels(ax, x_label, y_label, title, x, y, is_3d=True, z_label=z_label, z=z)
    #         self.set_limits(ax, x_lim, y_lim, x, y, data, is_3d=True, z_lim=z_lim, z=z)
    #         self.set_axes_visibility(ax, x_vis, y_vis, is_3d=True, z_vis=z_vis)
    #     axes = ax.axes
    #     if x_r:
    #         axes.invert_xaxis()
    #     if z_r:
    #         axes.invert_zaxis()
    #     if y_r:
    #         axes.invert_yaxis()

    # def population_activity_3d_raster(self, ax, z='s', x='x', y='y', y_label='spikes', z_label='time',
    #         reduction=1, **kwargs):
    #     data = self.get_3d_data(z='s', x='x', y='y', **kwargs)
    #     d = data[z]
    #     d = d.reshape(d.shape[0],-1)
    #     xe,ye = np.where(d)
    #     ye = np.array(ye)
    #     ye = ye//reduction
    #     xd = (np.array(ye)//data[z].shape[0])
    #     yd = (np.array(ye)%data[z].shape[1])
    #     zd = np.array(xe)
    #     self.scatter_3d(ax, data={z:zd, x:xd, y:yd}, z=z, x=x, y=y, z_label=z_label, **kwargs)

    # def surface_3d(self, ax, z='z', x='x', y='y', data=None, monitor=None, additive=False,
    #         title=None, x_label=None, y_label=None, z_label=None,
    #         x_vis=True, y_vis=True, z_vis=True,
    #         x_r=False, y_r=False, z_r=False,
    #         x_lim=None, y_lim=None, z_lim=None, **kwargs):
    #     if data is None:
    #         data = {}
    #     monitor = self.get_monitor(monitor)
    #     if z==None: z = str(ax)
    #     data = self.get_3d_data(z=z, y=y, x=x, data=data, monitor=monitor)
    #     Z = data[z].numpy()
    #     X = np.arange(Z.shape[0])
    #     Y = np.arange(Z.shape[1])
    #     X, Y = np.meshgrid(X, Y)
    #     ax = self.get_ax(ax)
    #     surf = ax.plot_surface(X, Y, Z, **kwargs)
    #     if not additive:
    #         self.set_labels(ax, x_label, y_label, title, x, y, is_3d=True, z_label=z_label, z=z)
    #         self.set_limits(ax, x_lim, y_lim, x, y, data, is_3d=True, z_lim=z_lim, z=z)
    #         self.set_axes_visibility(ax, x_vis, y_vis, is_3d=True, z_vis=z_vis)
    #     axes = ax.axes
    #     if x_r:
    #         axes.invert_xaxis()
    #     if z_r:
    #         axes.invert_zaxis()
    #     if y_r:
    #         axes.invert_yaxis()
    #     return surf