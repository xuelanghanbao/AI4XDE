import numpy as np
import deepxde as dde
import deepxde.backend as bkd
from .PDECases import PDECases
from abc import abstractmethod

class WaveCase1D(PDECases):
    def __init__(self, 
                 name,
                 c,
                 interval,
                 time_interval,
                 NumDomain=2000,
                 use_output_transform=False,
                 inverse=False,
                 layer_size = [2] + [32] * 3 + [1],
                 activation = 'sin',
                 initializer = 'Glorot uniform'):
        self.c = c
        self.interval = interval
        self.time_interval = time_interval
        super().__init__(name=name, NumDomain=NumDomain, use_output_transform=use_output_transform, inverse=inverse, layer_size=layer_size, activation=activation, initializer=initializer)

    @abstractmethod
    def gen_data(self):
        pass

    def gen_geomtime(self):
        geom = dde.geometry.Interval(self.interval[0], self.interval[1])
        timedomain = dde.geometry.TimeDomain(self.time_interval[0], self.time_interval[1])
        return dde.geometry.GeometryXTime(geom, timedomain)

    def gen_pde(self):
        def pde(x, y):
            dy_tt = dde.grad.hessian(y, x, i=1, j=1)
            dy_xx = dde.grad.hessian(y, x, i=0, j=0)
            return dy_tt - self.c**2 * dy_xx
        return pde
    
    def set_axes(self, axes):
        axes.set_xlim(self.time_interval[0], self.time_interval[1])
        axes.set_ylim(self.interval[0], self.interval[1])
        axes.set_xlabel('t')
        axes.set_ylabel('x')

    def plot_data(self, X, axes=None):
        from matplotlib import pyplot as plt
        if axes is None:
            fig, axes = plt.subplots()
        self.set_axes(axes)
        axes.scatter(X[:, 1], X[:, 0])
        return axes
    
    def plot_heatmap_at_axes(self, X, y, axes, title):
        axes.set_title(title)
        self.set_axes(axes)
        return axes.pcolormesh(X[:, 1].reshape(1000, 1000), X[:, 0].reshape(1000, 1000), y.reshape(1000, 1000), cmap='rainbow')
    
    def plot_result(self, solver, colorbar=[0,0,0]):
        from matplotlib import pyplot as plt
        X = np.array([[x, t] for x in np.linspace(self.interval[0], self.interval[1], 1000) for t in np.linspace(self.time_interval[0], self.time_interval[1], 1000)])
        y = self.sol(X)
        model_y = solver.model.predict(X)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axs = []
        axs.append(self.plot_heatmap_at_axes(X, y, axes=axes[0], title='Exact solution'))
        axs.append(self.plot_heatmap_at_axes(X, model_y, axes[1], title=solver.name))
        axs.append(self.plot_heatmap_at_axes(X, np.abs(model_y - y) , axes[2], title='Absolute error'))
        
        for needColorbar, ax, axe in zip(colorbar, axs, axes):
            if needColorbar:
                fig.colorbar(ax, ax=axe)
        plt.show()
        return fig, axes
    
class Wave_1D_STMsFFN(WaveCase1D):
    def __init__(self, 
                 NumDomain=360, 
                 layer_size=[2] + [100] * 3 + [1], 
                 activation='tanh', 
                 initializer='Glorot uniform'):
        super().__init__('Wave propagation with spatio-temporal multi-scale Fourier feature architecture',
                          c=2, 
                          interval=[0,1], 
                          time_interval=[0,1], 
                          NumDomain=NumDomain,
                          layer_size=layer_size,
                          activation=activation,
                          initializer=initializer)
    
    def sol(self, x):
        A = 2
        x, t = np.split(x, 2, axis=1)
        return np.sin(np.pi * x) * np.cos(self.c * np.pi * t) + np.sin(A * np.pi * x) * np.cos(
            A * self.c * np.pi * t
        )

    def gen_data(self):
        bc = dde.icbc.DirichletBC(self.geomtime, self.sol, lambda _, on_boundary: on_boundary)
        ic_1 = dde.icbc.IC(self.geomtime, self.sol, lambda _, on_initial: on_initial)
        # do not use dde.NeumannBC here, since `normal_derivative` does not work with temporal coordinate.
        ic_2 = dde.icbc.OperatorBC(
            self.geomtime,
            lambda x, y, _: dde.grad.jacobian(y, x, i=0, j=1),
            lambda x, _: np.isclose(x[1], 0),
        )
        return dde.data.TimePDE(self.geomtime, self.pde, [bc, ic_1, ic_2], num_domain=360, num_boundary=360, num_initial=360, num_test=10000, solution=self.sol )
    
    def gen_net(self, layer_size, activation, initializer):
        net = dde.nn.STMsFFN(
            layer_size, activation, initializer, sigmas_x=[1], sigmas_t=[1, 10]
        )
        net.apply_feature_transform(lambda x: (x - 0.5) * 2 * np.sqrt(3))
        return net
    
class Wave_1D_Hard_Boundary(WaveCase1D):
    """Case of Wave equation.
    Implementation of Wave equation example in paper https://arxiv.org/abs/2012.10047.
    References:
    https://github.com/PredictiveIntelligenceLab/MultiscalePINNs
    """
    def __init__(self, 
                 NumDomain=2000, 
                 layer_size=[2] + [100] * 5 + [1], 
                 activation='sin', 
                 initializer='Glorot normal'):
        super().__init__('Wave equation',
                          c=2, 
                          interval=[0,1], 
                          time_interval=[0,1], 
                          NumDomain=NumDomain,
                          use_output_transform=True,
                          layer_size=layer_size,
                          activation=activation,
                          initializer=initializer)
    
    def sol(self, x):
        return np.sin(np.pi * x[:, 0:1]) * np.cos(2 * np.pi * x[:, 1:2]) + 0.5 * np.sin(4 * np.pi * x[:, 0:1]) * np.cos(
            8 * np.pi * x[:, 1:2])

    def gen_data(self):
        return dde.data.TimePDE(self.geomtime, self.pde, [], num_domain=self.NumDomain, train_distribution='pseudo',
                            solution=self.sol, num_test=10000)
    
    def output_transform(self, x, y):
        x_in = x[:, 0:1]
        t_in = x[:, 1:2]
        return 20 * y * x_in * (1 - x_in) * t_in ** 2 + bkd.sin(np.pi * x_in) + 0.5 * bkd.sin(4 * np.pi * x_in)