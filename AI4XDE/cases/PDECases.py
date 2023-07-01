import numpy as np
import deepxde as dde
import deepxde.backend as bkd
from abc import ABC, abstractmethod

class PDECases(ABC):
    def __init__(self, 
                 name,
                 NumDomain=2000,
                 use_output_transform=False,
                 inverse=False,
                 layer_size = [2] + [32] * 3 + [1],
                 activation = 'tanh',
                 initializer = 'Glorot uniform'):
        self.name = name
        self.NumDomain = NumDomain
        self.use_output_transform = use_output_transform
        self.net = self.gen_net(layer_size, activation, initializer)
        self.pde = self.gen_pde()
        self.geomtime = self.gen_geomtime()
        self.data = self.gen_data()
           
    def gen_net(self, layer_size, activation, initializer):
        net = dde.nn.FNN(layer_size, activation, initializer)
        if self.use_output_transform:
            net.apply_output_transform(self.output_transform)
        return net
        
    @abstractmethod
    def gen_pde(self):
        pass

    @abstractmethod
    def gen_geomtime(self):
        pass

    @abstractmethod
    def gen_data(self):
        pass

    def gen_testdata(self):
        if callable(self.func):
            x = self.geomtime.random_points(self.NumDomain)
            y = self.func(x)
            return x, y
        else:
            raise Warning('You must rewrite one of func() and gen_testdata()')
    
    def output_transform(self, x, y):
        pass

    def set_axes(self, axes):
        pass

    def plot_data(self, X, axes=None):
        pass
    
    def plot_reslult(self, solver):
        pass

    def set_pde(self, pde):
        self.pde = pde
        self.data = self.gen_data()

class Burgers(PDECases):
    def __init__(self, 
                 NumDomain=2000, 
                 layer_size=[2] + [64] * 3 + [1], 
                 activation='tanh', 
                 initializer='Glorot normal'):
        super().__init__(name='Burgers', NumDomain=NumDomain, use_output_transform=True, layer_size=layer_size, activation=activation, initializer=initializer)
    
    def gen_pde(self):
        def pde(x, y):
            dy_x = dde.grad.jacobian(y, x, i=0, j=0)
            dy_t = dde.grad.jacobian(y, x, i=0, j=1)
            dy_xx = dde.grad.hessian(y, x, i=0, j=0)
            return dy_t + y * dy_x - 0.01 / np.pi * dy_xx
        return pde

    def gen_geomtime(self):
        geom = dde.geometry.Interval(-1, 1)
        timedomain = dde.geometry.TimeDomain(0, 1)
        return dde.geometry.GeometryXTime(geom, timedomain)
    
    def gen_data(self):
        return dde.data.TimePDE(self.geomtime, self.pde, [], num_domain=self.NumDomain, num_test=10000, train_distribution='pseudo')
    
    def gen_testdata(self):
        import os
        basepath = os.path.abspath(__file__)
        folder = os.path.dirname(os.path.dirname(basepath))
        data_path = os.path.join(folder, 'data/Burgers.npz')
        data = np.load(data_path)
        t, x, exact = data['t'], data['x'], data['usol'].T
        xx, tt = np.meshgrid(x, t)
        X = np.vstack((np.ravel(xx), np.ravel(tt))).T
        y = exact.flatten()[:, None]
        return X, y
    
    def output_transform(self, x, y):
        return -bkd.sin(np.pi * x[:, 0:1]) + (1 - x[:, 0:1] ** 2) * (x[:, 1:]) * y
    
    def set_axes(self, axes):
        axes.set_xlim(0, 1)
        axes.set_ylim(-1, 1)
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
        axes.pcolormesh(X[:, 1].reshape(100, 256), X[:, 0].reshape(100, 256), y.reshape(100, 256), cmap='rainbow')
    
    def plot_result(self, solver, colorbar=None):
        from matplotlib import pyplot as plt
        X, y = self.gen_testdata()
        model_y = solver.model.predict(X)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axs = []
        axs.append(self.plot_heatmap_at_axes(X, y, axes=axes[0], title='Exact solution'))
        axs.append(self.plot_heatmap_at_axes(X, model_y, axes[1], title=solver.name))
        axs.append(self.plot_heatmap_at_axes(X, np.abs(model_y - y) , axes[2], title='Absolute error'))
        
        if colorbar is not None:
            for needColorbar, ax, axe in zip(colorbar, axs, axes):
                if needColorbar:
                    fig.colorbar(ax, ax=axe)
        plt.show()
        return fig, axes
    

class AllenCahn(PDECases):
    """Case of Allen-Cahn equation.
    Implementation of Allen-Cahn equation example in paper https://arxiv.org/abs/2111.02801.
    """
    def __init__(self, 
                 NumDomain=2000, 
                 layer_size=[2] + [64] * 3 + [1], 
                 activation='tanh', 
                 initializer='Glorot normal'):
        super().__init__(name='AllenCahn', NumDomain=NumDomain, use_output_transform=True, layer_size=layer_size, activation=activation, initializer=initializer)

    def gen_pde(self):
        def pde(x, y):
            u = y
            du_xx = dde.grad.hessian(y, x, i=0, j=0)
            du_t = dde.grad.jacobian(y, x, j=1)
            return du_t - 0.001 * du_xx + 5 * (u ** 3 - u)
        return pde
    
    def gen_geomtime(self):
        geom = dde.geometry.Interval(-1, 1)
        timedomain = dde.geometry.TimeDomain(0, 1)
        return dde.geometry.GeometryXTime(geom, timedomain)
    
    def gen_data(self):
        return dde.data.TimePDE(self.geomtime, self.pde, [], num_domain=self.NumDomain, num_test=10000, train_distribution='pseudo')

    def gen_testdata(self):
        import os
        from scipy.io import loadmat
        basepath = os.path.abspath(__file__)
        folder = os.path.dirname(os.path.dirname(basepath))
        data_path = os.path.join(folder, 'data/Allen_Cahn.mat')
        data = loadmat(data_path)
        t = data['t']
        x = data['x']
        u = data['u']
        dt = dx = 0.01
        xx, tt = np.meshgrid(x, t)
        X = np.vstack((np.ravel(xx), np.ravel(tt))).T
        y = u.flatten()[:, None]
        return X, y
    
    def output_transform(self, x, y):
        x_in = x[:, 0:1]
        t_in = x[:, 1:2]
        return t_in * (1 + x_in) * (1 - x_in) * y + bkd.square(x_in) * bkd.cos(np.pi * x_in)
    
    def set_axes(self, axes):
        axes.set_xlim(0, 1)
        axes.set_ylim(-1, 1)
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
        axes.pcolormesh(X[:, 1].reshape(101, 201), X[:, 0].reshape(101, 201), y.reshape(101, 201), cmap='rainbow')
    
    def plot_result(self, solver, colorbar=None):
        from matplotlib import pyplot as plt
        X, y = self.gen_testdata()
        model_y = solver.model.predict(X)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axs = []
        axs.append(self.plot_heatmap_at_axes(X, y, axes=axes[0], title='Exact solution'))
        axs.append(self.plot_heatmap_at_axes(X, model_y, axes[1], title=solver.name))
        axs.append(self.plot_heatmap_at_axes(X, np.abs(model_y - y) , axes[2], title='Absolute error'))
        
        if colorbar is not None:
            for needColorbar, ax, axe in zip(colorbar, axs, axes):
                if needColorbar:
                    fig.colorbar(ax, ax=axe)
        plt.show()
        return fig, axes
    
class Diffusion(PDECases):
    def __init__(self, 
                 NumDomain=2000, 
                 layer_size=[2] + [32] * 3 + [1], 
                 activation='tanh', 
                 initializer='Glorot normal'):
        super().__init__(name='Diffusion', NumDomain=NumDomain, use_output_transform=True, layer_size=layer_size, activation=activation, initializer=initializer)

    def gen_pde(self):
        def pde(x, y):
            dy_t = dde.grad.jacobian(y, x, j=1)
            dy_xx = dde.grad.hessian(y, x, j=0)
            return (
                    dy_t
                    - dy_xx
                    +  bkd.exp(-x[:, 1:])
                    * (bkd.sin(np.pi * x[:, 0:1]) - np.pi ** 2 * bkd.sin(np.pi * x[:, 0:1]))
            )
        return pde
    
    def gen_geomtime(self):
        geom = dde.geometry.Interval(-1, 1)
        timedomain = dde.geometry.TimeDomain(0, 1)
        return dde.geometry.GeometryXTime(geom, timedomain)
    
    def func(self, x):
        return np.sin(np.pi * x[:, 0:1]) * np.exp(-x[:, 1:])
    
    def gen_data(self):
        return dde.data.TimePDE(self.geomtime, self.pde, [], num_domain=self.NumDomain, train_distribution='pseudo',
                            solution=self.func, num_test=10000)
    
    def output_transform(self, x, y):
        return bkd.sin(np.pi * x[:, 0:1]) + (1 - x[:, 0:1] ** 2) * (x[:, 1:]) * y
    
    def set_axes(self, axes):
        axes.set_xlim(0, 1)
        axes.set_ylim(-1, 1)
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
    
    def plot_result(self, solver, colorbar=None):
        from matplotlib import pyplot as plt
        X = np.array([[x, t] for x in np.linspace(-1, 1, 1000) for t in np.linspace(0, 1, 1000)])
        y = self.func(X)
        model_y = solver.model.predict(X)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axs = []
        axs.append(self.plot_heatmap_at_axes(X, y, axes=axes[0], title='Exact solution'))
        axs.append(self.plot_heatmap_at_axes(X, model_y, axes[1], title=solver.name))
        axs.append(self.plot_heatmap_at_axes(X, np.abs(model_y - y) , axes[2], title='Absolute error'))
        
        if colorbar is not None:
            for needColorbar, ax, axe in zip(colorbar, axs, axes):
                if needColorbar:
                    fig.colorbar(ax, ax=axe)
        plt.show()
        return fig, axes
        
class Wave(PDECases):
    """Case of Wave equation.
    Implementation of Wave equation example in paper https://arxiv.org/abs/2012.10047.
    References:
    https://github.com/PredictiveIntelligenceLab/MultiscalePINNs
    """
    def __init__(self, 
                 NumDomain=2000, 
                 layer_size=[2] + [100] * 5 + [1], 
                 activation='tanh', 
                 initializer='Glorot normal'):
        super().__init__(name='Wave', NumDomain=NumDomain, use_output_transform=True, layer_size=layer_size, activation=activation, initializer=initializer)

    def gen_pde(self):
        def pde(x, y):
            dy_tt = dde.grad.hessian(y, x, i=1, j=1)
            dy_xx = dde.grad.hessian(y, x, i=0, j=0)
            return dy_tt - 4.0 * dy_xx
        return pde
    
    def gen_geomtime(self):
        geom = dde.geometry.Interval(0, 1)
        timedomain = dde.geometry.TimeDomain(0, 1)
        return dde.geometry.GeometryXTime(geom, timedomain)
    
    def func(self, x):
        return np.sin(np.pi * x[:, 0:1]) * np.cos(2 * np.pi * x[:, 1:2]) + 0.5 * np.sin(4 * np.pi * x[:, 0:1]) * np.cos(
            8 * np.pi * x[:, 1:2])
    
    def gen_data(self):
        return dde.data.TimePDE(self.geomtime, self.pde, [], num_domain=self.NumDomain, train_distribution='pseudo',
                            solution=self.func, num_test=10000)
    
    def output_transform(self, x, y):
        x_in = x[:, 0:1]
        t_in = x[:, 1:2]
        return 20 * y * x_in * (1 - x_in) * t_in ** 2 + bkd.sin(np.pi * x_in) + 0.5 * bkd.sin(4 * np.pi * x_in)
    
    def set_axes(self, axes):
        tl = self.geomtime.timedomain.t0
        tr = self.geomtime.timedomain.t1
        xl = self.geomtime.geometry.l
        xr = self.geomtime.geometry.r
        axes.set_xlim(tl, tr)
        axes.set_ylim(xl, xr)
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
    
    def plot_result(self, solver, colorbar=None):
        from matplotlib import pyplot as plt
        tl = self.geomtime.timedomain.t0
        tr = self.geomtime.timedomain.t1
        xl = self.geomtime.geometry.l
        xr = self.geomtime.geometry.r
        X = np.array([[x, t] for x in np.linspace(xl, xr, 1000) for t in np.linspace(tl, tr, 1000)])
        y = self.func(X)
        model_y = solver.model.predict(X)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axs = []
        axs.append(self.plot_heatmap_at_axes(X, y, axes=axes[0], title='Exact solution'))
        axs.append(self.plot_heatmap_at_axes(X, model_y, axes[1], title=solver.name))
        axs.append(self.plot_heatmap_at_axes(X, np.abs(model_y - y) , axes[2], title='Absolute error'))
        
        if colorbar is not None:
            for needColorbar, ax, axe in zip(colorbar, axs, axes):
                if needColorbar:
                    fig.colorbar(ax, ax=axe)
        plt.show()
        return fig, axes
    
class Diffusion_Reaction_Inverse(PDECases):
    def __init__(self,
                 NumDomain=2000, 
                 use_output_transform=False, 
                 layer_size=[1, [20, 20], [20, 20], [20, 20], 2], 
                 activation='tanh', 
                 initializer='Glorot uniform'):
        self.res = self.gen_res()
        self.metrics = self.gen_metrics()
        super().__init__(name='Diffusion_Reaction_Inverse', NumDomain=NumDomain, use_output_transform=use_output_transform, inverse=True, layer_size=layer_size, activation=activation, initializer=initializer)

    def sol(self,x):
        return self.res.sol(x)[0]

    def gen_res(self):
        from scipy.integrate import solve_bvp
        def k(x):
            return 0.1 + np.exp(-0.5 * (x - 0.5) ** 2 / 0.15 ** 2)

        def fun(x, y):
            return np.vstack((y[1], 100 * (k(x) * y[0] + np.sin(2 * np.pi * x))))

        def bc(ya, yb):
            return np.array([ya[0], yb[0]])

        a = np.linspace(0, 1, 1000)
        b = np.zeros((2, a.size))

        res = solve_bvp(fun, bc, a, b)
        return res
    
    def get_metrics(self,model):
        xx = np.linspace(0, 1, 1001)[:, None]
        def k(x):
            return 0.1 + np.exp(-0.5 * (x - 0.5) ** 2 / 0.15 ** 2)
        def l2_u(_, __):
            return dde.metrics.l2_relative_error(self.sol(xx), model.predict(xx)[:, 0:1])
        def l2_k(_, __):
            return dde.metrics.l2_relative_error(k(xx), model.predict(xx)[:, 1:2])
        return [l2_u, l2_k]

    def gen_pde(self):
        def pde(x, y):
            u = y[:, 0:1]
            k = y[:, 1:2]
            du_xx = dde.grad.hessian(y, x, component=0)
            return 0.01 * du_xx - k * u - bkd.sin(2 * np.pi * x)
        return pde
    
    def gen_net(self, layer_size, activation, initializer):
        return dde.nn.PFNN(layer_size, activation, initializer)
    
    def gen_geomtime(self):
        return dde.geometry.Interval(0, 1)
    
    def gen_data(self):
        def gen_traindata(num):
            xvals = np.linspace(0, 1, num)
            yvals = self.sol(xvals)

            return np.reshape(xvals, (-1, 1)), np.reshape(yvals, (-1, 1))
        
        ob_x, ob_u = gen_traindata(8)
        observe_u = dde.PointSetBC(ob_x, ob_u, component=0)
        bc = dde.DirichletBC(self.geomtime, self.sol, lambda _, on_boundary: on_boundary, component=0)
        return dde.data.PDE(self.geomtime, self.pde, bcs=[bc, observe_u], num_domain=self.NumDomain-2, num_boundary=2,
                        train_distribution="pseudo", num_test=1000)
    
    

class A_Simple_ODE(PDECases):
    def __init__(self, 
                 NumDomain=2000, 
                 layer_size=[1] + [64] * 3 + [2], 
                 activation='tanh', 
                 initializer='Glorot normal'):
        super().__init__(name='A Simple ODE', NumDomain=NumDomain, use_output_transform=False, layer_size=layer_size, activation=activation, initializer=initializer)
    
    def gen_pde(self):
        def ode(x, y):
            y1, y2 = y[:, 0:1], y[:, 1:]
            dy1_x = dde.grad.jacobian(y, x, i=0)
            dy2_x = dde.grad.jacobian(y, x, i=1)
            return [dy1_x - y2, dy2_x + y1]
        return ode

    def gen_geomtime(self):
        geom = dde.geometry.TimeDomain(0, 10*np.pi)
        return geom
    
    def gen_data(self):
        def boundary(_, on_initial):
            return on_initial
        ic1 = dde.icbc.IC(self.geomtime, lambda x: 0, boundary, component=0)
        ic2 = dde.icbc.IC(self.geomtime, lambda x: 1, boundary, component=1)
        return dde.data.PDE(self.geomtime, self.pde, [ic1, ic2], num_domain=self.NumDomain, num_boundary=2, solution=self.func, num_test=10000, train_distribution='pseudo')
    
    def func(self, x):
        return np.hstack((np.sin(x), np.cos(x)))
    
    def plot_result(self, model, axes=None, exact=False):
        from matplotlib import pyplot as plt
        xx = np.linspace(-np.pi/2, np.pi/2, 1001)[:, None]
        if axes is None:
            fig, axes = plt.subplots()
        if exact:
            axes.plot(xx, self.func(xx), label='Exact')
        axes.plot(xx, model.predict(xx), label='Prediction')
        axes.legend()
        axes.set_xlabel('x')
        axes.set_ylabel('y')
        axes.set_title(self.name)
        return axes
    
class LotkaVolterra(PDECases):
    def __init__(self, 
                 NumDomain=3000, 
                 layer_size=[1] + [64] * 6 + [2], 
                 activation='tanh', 
                 initializer='Glorot normal'):
        super().__init__(name='Lotka-Volterra', NumDomain=NumDomain, use_output_transform=False, layer_size=layer_size, activation=activation, initializer=initializer)
        self.ub = 200
        self.rb = 20
    
    def gen_pde(self):
        def ode_system(x, y):
            r = y[:, 0:1]
            p = y[:, 1:2]
            dr_t = dde.grad.jacobian(y, x, i=0)
            dp_t = dde.grad.jacobian(y, x, i=1)
            return [
                dr_t - 1 / self.ub * self.rb * (2.0 * self.ub * r - 0.04 * self.ub * r * self.ub * p),
                dp_t - 1 / self.ub * self.rb * (0.02 * r * self.ub * p * self.ub - 1.06 * p * self.ub),
            ]
        return ode_system
    
    def gen_net(self, layer_size, activation, initializer):
        net = dde.nn.FNN(layer_size, activation, initializer)
        net.apply_feature_transform(self.input_transform)
        net.apply_output_transform(self.output_transform)
        return net

    def output_transform(self, t, y):
        y1 = y[:, 0:1]
        y2 = y[:, 1:2]
        return bkd.concat([y1 * bkd.tanh(t) + 100 / self.ub, y2 * bkd.tanh(t) + 15 / self.ub], axis=1)
        
    def input_transform(self,t):
        return bkd.concat(
            (
                bkd.sin(t),
            ),
            axis=1,
        )

    def gen_geomtime(self):
        geom = dde.geometry.TimeDomain(0, 1.0)
        return geom
    
    def gen_data(self):
        return dde.data.PDE(self.geomtime, self.pde, [], num_domain=self.NumDomain, num_boundary=2, num_test=3000)
    
    def gen_testdata(self):
        from scipy import integrate
        def func(t, r):
            x, y = r
            dx_t = 1 / self.ub * self.rb * (2.0 * self.ub * x - 0.04 * self.ub * x * self.ub * y)
            dy_t = 1 / self.ub * self.rb * (0.02 * self.ub * x * self.ub * y - 1.06 * self.ub * y)
            return dx_t, dy_t
        t = np.linspace(0, 1, 100)

        sol = integrate.solve_ivp(func, (0, 10), (100 / self.ub, 15 / self.ub), t_eval=t)
        x_true, y_true = sol.y
        x_true = x_true.reshape(100, 1)
        y_true = y_true.reshape(100, 1)

        return x_true, y_true
    
    def plot_result(self, solver, axes=None, exact=False):
        from matplotlib import pyplot as plt
        X,y = self.gen_testdata()
        if axes is None:
            fig, axes = plt.subplots()
        if exact:
            axes.plot(X, y, label='Exact')
        axes.plot(X, solver.model.predict(X), label='Prediction')
        axes.legend()
        axes.set_xlabel('t')
        axes.set_ylabel('population')
        axes.set_title(self.name)
        return axes
            
    