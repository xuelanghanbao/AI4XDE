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
        if callable(self.sol):
            x = self.geomtime.uniform_points(self.NumDomain)
            y = self.sol(x)
            return x, y
        else:
            raise Warning('You must rewrite one of sol() and gen_testdata()')
    
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
        return axes.pcolormesh(X[:, 1].reshape(100, 256), X[:, 0].reshape(100, 256), y.reshape(100, 256), cmap='rainbow')
    
    def plot_result(self, solver, colorbar=[0,0,0]):
        from matplotlib import pyplot as plt
        X, y = self.gen_testdata()
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
        return axes.pcolormesh(X[:, 1].reshape(101, 201), X[:, 0].reshape(101, 201), y.reshape(101, 201), cmap='rainbow')
    
    def plot_result(self, solver, colorbar=[0,0,0]):
        from matplotlib import pyplot as plt
        X, y = self.gen_testdata()
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
    
    def sol(self, x):
        return np.sin(np.pi * x[:, 0:1]) * np.exp(-x[:, 1:])
    
    def gen_data(self):
        return dde.data.TimePDE(self.geomtime, self.pde, [], num_domain=self.NumDomain, train_distribution='pseudo',
                            solution=self.sol, num_test=10000)
    
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
    
    def plot_result(self, solver, colorbar=[0,0,0]):
        from matplotlib import pyplot as plt
        X = np.array([[x, t] for x in np.linspace(-1, 1, 1000) for t in np.linspace(0, 1, 1000)])
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
    
    def plot_result(self, solver, colorbar=[0,0,0]):
        from matplotlib import pyplot as plt
        tl = self.geomtime.timedomain.t0
        tr = self.geomtime.timedomain.t1
        xl = self.geomtime.geometry.l
        xr = self.geomtime.geometry.r
        X = np.array([[x, t] for x in np.linspace(xl, xr, 1000) for t in np.linspace(tl, tr, 1000)])
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
        return dde.data.PDE(self.geomtime, self.pde, [ic1, ic2], num_domain=self.NumDomain, num_boundary=2, solution=self.sol, num_test=10000, train_distribution='pseudo')
    
    def sol(self, x):
        return np.hstack((np.sin(x), np.cos(x)))
    
    def plot_result(self, model, axes=None, exact=False):
        from matplotlib import pyplot as plt
        xx = np.linspace(-np.pi/2, np.pi/2, 1001)[:, None]
        if axes is None:
            fig, axes = plt.subplots()
        if exact:
            axes.plot(xx, self.sol(xx), label='Exact')
        axes.plot(xx, model.predict(xx), '--', label='Prediction')
        axes.legend()
        axes.set_xlabel('x')
        axes.set_ylabel('y')
        axes.set_title(self.name)
        return fig, axes
    
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
        def sol(t, r):
            x, y = r
            dx_t = 1 / self.ub * self.rb * (2.0 * self.ub * x - 0.04 * self.ub * x * self.ub * y)
            dy_t = 1 / self.ub * self.rb * (0.02 * self.ub * x * self.ub * y - 1.06 * self.ub * y)
            return dx_t, dy_t
        t = np.linspace(0, 1, 100)

        sol = integrate.solve_ivp(sol, (0, 10), (100 / self.ub, 15 / self.ub), t_eval=t)
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
        axes.plot(X, solver.model.predict(X), '--', label='Prediction')
        axes.legend()
        axes.set_xlabel('t')
        axes.set_ylabel('population')
        axes.set_title(self.name)
        return fig, axes
    
class SecondOrderODE(PDECases):
    def __init__(self, 
                 NumDomain=16, 
                 layer_size=[1] + [50] * 3 + [1], 
                 activation='tanh', 
                 initializer='Glorot normal'):
        super().__init__(name='Second Order ODE', NumDomain=NumDomain, use_output_transform=False, layer_size=layer_size, activation=activation, initializer=initializer)
        self.ub = 200
        self.rb = 20
    
    def gen_pde(self):
        def ode_system(t, y):
            dy_dt = dde.grad.jacobian(y, t)
            d2y_dt2 = dde.grad.hessian(y, t)
            return d2y_dt2 - 10 * dy_dt + 9 * y - 5 * t
        return ode_system
    
    def sol(self, t):
        return 50 / 81 + t * 5 / 9 - 2 * np.exp(t) + (31 / 81) * np.exp(9 * t)

    def gen_geomtime(self):
        geom = dde.geometry.TimeDomain(0, 0.25)
        return geom
    
    def gen_data(self):
        def boundary_l(t, on_initial):
            return on_initial and np.isclose(t[0], 0)
        def bc_func2(inputs, outputs, X):
            return dde.grad.jacobian(outputs, inputs, i=0, j=None) - 2
        ic1 = dde.icbc.IC(self.geomtime, lambda x: -1, lambda _, on_initial: on_initial)
        ic2 = dde.icbc.OperatorBC(self.geomtime, bc_func2, boundary_l)
        return dde.data.TimePDE(self.geomtime, self.pde, [ic1, ic2], self.NumDomain, 2, solution=self.sol, num_test=500)
    
    def plot_result(self, solver, axes=None, exact=False):
        from matplotlib import pyplot as plt
        X,y = self.gen_testdata()
        if axes is None:
            fig, axes = plt.subplots()
        if exact:
            axes.plot(X, y, label='Exact')
        axes.plot(X, solver.model.predict(X), '--', label='Prediction')
        axes.legend()
        axes.set_xlabel('t')
        axes.set_ylabel('y')
        axes.set_title(self.name)
        return fig, axes

class Laplace_disk(PDECases):
    def __init__(self, 
                 NumDomain=2540, 
                 layer_size=[2] + [20] * 3 + [1], 
                 activation='tanh', 
                 initializer='Glorot normal'):
        super().__init__(name='Laplace equation on a disk', NumDomain=NumDomain, use_output_transform=False, layer_size=layer_size, activation=activation, initializer=initializer)
    
    def gen_pde(self):
        def pde(x, y):
            dy_r = dde.grad.jacobian(y, x, i=0, j=0)
            dy_rr = dde.grad.hessian(y, x, i=0, j=0)
            dy_thetatheta = dde.grad.hessian(y, x, i=1, j=1)
            return x[:, 0:1] * dy_r + x[:, 0:1] ** 2 * dy_rr + dy_thetatheta
        return pde
    
    def sol(self, x):
        r, theta = x[:, 0:1], x[:, 1:]
        return r * np.cos(theta)

    def gen_geomtime(self):
        return dde.geometry.Rectangle(xmin=[0, 0], xmax=[1, 2 * np.pi])
    
    def gen_data(self):
        bc_rad = dde.icbc.DirichletBC(self.geomtime, lambda x: np.cos(x[:, 1:2]), lambda x, on_boundary: on_boundary and np.isclose(x[0], 1))
        return dde.data.TimePDE(self.geomtime, self.pde, bc_rad, self.NumDomain, 80, solution=self.sol)
    
    def gen_net(self, layer_size, activation, initializer):
        def feature_transform(x):
            return bkd.concat([x[:, 0:1] * bkd.sin(x[:, 1:2]), x[:, 0:1] * bkd.cos(x[:, 1:2])], axis=1)
        net = dde.nn.FNN(layer_size, activation, initializer)
        net.apply_feature_transform(feature_transform)
        return net
    
    def set_axes(self, axes):
        axes.set_xlim(-1, 1)
        axes.set_ylim(-1, 1)
        axes.set_xlabel('x1')
        axes.set_ylabel('x2')

    def plot_data(self, X, axes=None):
        from matplotlib import pyplot as plt
        if axes is None:
            fig, axes = plt.subplots()
        self.set_axes(axes)
        X = np.array([[ x[0]*np.cos(x[1]), x[0]*np.sin(x[1])] for x in X])
        axes.scatter(X[:, 0], X[:, 1])
        return axes
    
    def plot_heatmap_at_axes(self, X, y, axes, title):
        axes.set_title(title)
        self.set_axes(axes)
        return axes.pcolormesh(X[:, 0].reshape(1000, 1000), X[:, 1].reshape(1000, 1000), y.reshape(1000, 1000), cmap='rainbow')
    
    def plot_result(self, solver, colorbar=[0,0,0]):
        from matplotlib import pyplot as plt
        X = np.array([[x1, x2] for x1 in np.linspace(0, 1, 1000) for x2 in np.linspace(0, 2*np.pi, 1000)])
        X_R = np.array([[ x[0]*np.cos(x[1]), x[0]*np.sin(x[1])] for x in X])
        y = self.sol(X)
        model_y = solver.model.predict(X)

        y[self.geomtime.inside(X)==0] = np.nan
        model_y[self.geomtime.inside(X)==0] = np.nan

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axs = []
        axs.append(self.plot_heatmap_at_axes(X_R, y, axes=axes[0], title='Exact solution'))
        axs.append(self.plot_heatmap_at_axes(X_R, model_y, axes[1], title=solver.name))
        axs.append(self.plot_heatmap_at_axes(X_R, np.abs(model_y - y) , axes[2], title='Absolute error'))

        for needColorbar, ax, axe in zip(colorbar, axs, axes):
            if needColorbar:
                fig.colorbar(ax, ax=axe)
        
        plt.show()
        return fig, axes
    
class Helmholtz(PDECases):
    def __init__(self, 
                 NumDomain=2540, 
                 hard_constraint=False,
                 layer_size=[2] + [150] * 3 + [1], 
                 activation='sin', 
                 initializer='Glorot uniform'):
        self.n = 2
        self.k0 = 2*np.pi*self.n
        self.hard_constraint = hard_constraint
        super().__init__(name='Helmholtz equation over a 2D square domain', NumDomain=NumDomain, use_output_transform=False, layer_size=layer_size, activation=activation, initializer=initializer)
    
    def gen_pde(self):
        def pde(x, y):
            dy_xx = dde.grad.hessian(y, x, i=0, j=0)
            dy_yy = dde.grad.hessian(y, x, i=1, j=1)

            f = self.k0 ** 2 * bkd.sin(self.k0 * x[:, 0:1]) * bkd.sin(self.k0 * x[:, 1:2])
            return -dy_xx - dy_yy - self.k0 ** 2 * y - f
        return pde
    
    def sol(self, x):
        return np.sin(self.k0 * x[:, 0:1]) * np.sin(self.k0 * x[:, 1:2])

    def gen_geomtime(self):
        return dde.geometry.Rectangle([0, 0], [1, 1])
    
    def gen_data(self):
        if self.hard_constraint == True:
            bc = []
        else:
            bc = dde.icbc.DirichletBC(self.geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)

        precision_train = 10
        precision_test = 30
        wave_len = 1 / self.n

        hx_train = wave_len / precision_train
        nx_train = int(1 / hx_train)

        hx_test = wave_len / precision_test
        nx_test = int(1 / hx_test)
        return dde.data.PDE(self.geomtime, self.pde, bc, num_domain=nx_train ** 2, num_boundary=4 * nx_train, solution=self.sol,num_test=nx_test ** 2)
    
    def set_axes(self, axes):
        axes.set_xlim(0, 1)
        axes.set_ylim(0, 1)
        axes.set_xlabel('x1')
        axes.set_ylabel('x2')

    def plot_data(self, X, axes=None):
        from matplotlib import pyplot as plt
        if axes is None:
            fig, axes = plt.subplots()
        self.set_axes(axes)
        axes.scatter(X[:, 0], X[:, 1])
        return axes
    
    def plot_heatmap_at_axes(self, X, y, axes, title):
        axes.set_title(title)
        self.set_axes(axes)
        return axes.pcolormesh(X[:, 0].reshape(1000, 1000), X[:, 1].reshape(1000, 1000), y.reshape(1000, 1000), cmap='rainbow')
    
    def plot_result(self, solver, colorbar=[0,0,0]):
        from matplotlib import pyplot as plt
        X = np.array([[x1, x2] for x1 in np.linspace(0, 1, 1000) for x2 in np.linspace(0, 1, 1000)])
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
    
class Helmholtz_Hole(PDECases):
    def __init__(self, 
                 precision_train=15,
                 precision_test=30, 
                 hard_constraint=False,
                 layer_size=[2] + [350] * 3 + [1], 
                 activation='sin', 
                 initializer='Glorot uniform'):
        self.n = 1
        self.k0 = 2*np.pi*self.n
        self.hard_constraint = hard_constraint
        self.NumDomain, self.NumBoundary, self.NumTest = self.get_NumDomain(precision_train, precision_test)

        R = 1/4
        length = 1
        self.inner = dde.geometry.Disk([0, 0], R)
        self.outer = dde.geometry.Rectangle([-length / 2, -length / 2], [length / 2, length / 2])

        super().__init__(name='Helmholtz equation over a 2D square domain with a hole', NumDomain=self.NumDomain, use_output_transform=False, layer_size=layer_size, activation=activation, initializer=initializer)
    
    def get_NumDomain(self, precision_train, precision_test):
        wave_len = 1 / self.n

        hx_train = wave_len / precision_train
        nx_train = int(1 / hx_train)

        hx_test = wave_len / precision_test
        nx_test = int(1 / hx_test)

        num_domain = nx_train ** 2
        num_boundary = 4 * nx_train
        num_test = nx_test ** 2
        return num_domain, num_boundary, num_test

    def gen_pde(self):
        def pde(x, y):
            dy_xx = dde.grad.hessian(y, x, i=0, j=0)
            dy_yy = dde.grad.hessian(y, x, i=1, j=1)

            f = self.k0 ** 2 * bkd.sin(self.k0 * x[:, 0:1]) * bkd.sin(self.k0 * x[:, 1:2])
            return -dy_xx - dy_yy - self.k0 ** 2 * y - f
        return pde
    
    def sol(self, x):
        return np.sin(self.k0 * x[:, 0:1]) * np.sin(self.k0 * x[:, 1:2])

    def gen_geomtime(self):
        return self.outer - self.inner
    
    def gen_data(self):
        def neumann(x):
            grad = np.array(
                [
                    self.k0 * np.cos(self.k0 * x[:, 0:1]) * np.sin(self.k0 * x[:, 1:2]),
                    self.k0 * np.sin(self.k0 * x[:, 0:1]) * np.cos(self.k0 * x[:, 1:2]),
                ]
            )

            normal = -self.inner.boundary_normal(x)
            normal = np.array([normal]).T
            result = np.sum(grad * normal, axis=0)
            return result
        def boundary_inner(x, on_boundary):
            return on_boundary and self.inner.on_boundary(x)
        def boundary_outer(x, on_boundary):
            return on_boundary and self.outer.on_boundary(x)

        bc_inner = dde.icbc.NeumannBC(self.geomtime, neumann, boundary_inner)
        bc_outer = dde.icbc.DirichletBC(self.geomtime, self.sol, boundary_outer)
        
        return dde.data.PDE(self.geomtime, self.pde, [bc_inner, bc_outer], num_domain=self.NumDomain, num_boundary=self.NumBoundary, solution=self.sol,num_test=self.NumTest)
    
    def set_axes(self, axes):
        axes.set_xlim(-1/2, 1/2)
        axes.set_ylim(-1/2, 1/2)
        axes.set_xlabel('x1')
        axes.set_ylabel('x2')

    def plot_data(self, X, axes=None):
        from matplotlib import pyplot as plt
        if axes is None:
            fig, axes = plt.subplots()
        self.set_axes(axes)
        axes.scatter(X[:, 0], X[:, 1])
        return axes
    
    def plot_heatmap_at_axes(self, X, y, axes, title):
        axes.set_title(title)
        self.set_axes(axes)
        return axes.pcolormesh(X[:, 0].reshape(1000, 1000), X[:, 1].reshape(1000, 1000), y.reshape(1000, 1000), cmap='rainbow')
    
    def plot_result(self, solver, colorbar=[0,0,0]):
        from matplotlib import pyplot as plt
        X = np.array([[x1, x2] for x1 in np.linspace(-1/2, 1/2, 1000) for x2 in np.linspace(-1/2, 1/2, 1000)])
        y = self.sol(X)
        y[ self.geomtime.inside(X) == 0 ] = np.nan
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
    
class Helmholtz_Sound_hard_Absorbing(PDECases):
    def __init__(self, 
                 hard_constraint=False,
                 layer_size=[2] + [350] * 3 + [2], 
                 activation='tanh', 
                 initializer='Glorot uniform'):
        
        self.R = np.pi / 4
        self.length = 2 * np.pi
        self.inner = dde.geometry.Disk([0, 0], self.R)
        self.outer = dde.geometry.Rectangle([-self.length / 2, -self.length / 2], [self.length / 2, self.length / 2])

        self.n = 1
        self.k0 = 2*np.pi*self.n
        self.hard_constraint = hard_constraint
        self.NumDomain, self.NumBoundary, self.NumTest = self.get_NumDomain()

        super().__init__(name='Helmholtz sound-hard scattering problem with absorbing boundary conditions', NumDomain=self.NumDomain, use_output_transform=False, layer_size=layer_size, activation=activation, initializer=initializer)
    
    def get_NumDomain(self):
        wave_len = 2 * np.pi / self.k0
        n_wave = 20
        h_elem = wave_len / n_wave
        nx = int(self.length / h_elem)

        num_domain = nx ** 2
        num_boundary = 8 * nx
        num_test = 5 * nx ** 2
        return num_domain, num_boundary, num_test

    def gen_pde(self):
        def pde(x, y):
            y0, y1 = y[:, 0:1], y[:, 1:2]

            y0_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
            y0_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)

            y1_xx = dde.grad.hessian(y, x,component=1, i=0, j=0)
            y1_yy = dde.grad.hessian(y, x,component=1, i=1, j=1)

            return [-y0_xx - y0_yy - self.k0 ** 2 * y0,
                    -y1_xx - y1_yy - self.k0 ** 2 * y1]
        return pde
    
    def sound_hard_circle_deepxde(self, k0, a, points):
        from scipy.special import jv, hankel1
        fem_xx = points[:, 0:1]
        fem_xy = points[:, 1:2]
        r = np.sqrt(fem_xx * fem_xx + fem_xy * fem_xy)
        theta = np.arctan2(fem_xy, fem_xx)
        npts = np.size(fem_xx, 0)
        n_terms = int(30 + (k0 * a)**1.01)
        u_sc = np.zeros((npts), dtype=np.complex128)
        for n in range(-n_terms, n_terms):
            bessel_deriv = jv(n-1, k0*a) - n/(k0*a) * jv(n, k0*a)
            hankel_deriv = n/(k0*a)*hankel1(n, k0*a) - hankel1(n+1, k0*a)
            u_sc += (-(1j)**(n) * (bessel_deriv/hankel_deriv) * hankel1(n, k0*r) * \
                np.exp(1j*n*theta)).ravel()
        return u_sc

    def sol(self, x):
        result = self.sound_hard_circle_deepxde(self.k0, self.R, x).reshape((x.shape[0],1))
        real = np.real(result)
        imag = np.imag(result)
        return np.hstack((real, imag))

    def gen_geomtime(self):
        return self.outer - self.inner
    
    def gen_data(self):
        def boundary_inner(x, on_boundary):
            return on_boundary and self.inner.on_boundary(x)
        def boundary_outer(x, on_boundary):
            return on_boundary and self.outer.on_boundary(x)
        def func0_inner(x):
            normal = -self.inner.boundary_normal(x)
            g = 1j * self.k0 * np.exp(1j * self.k0 * x[:, 0:1]) * normal[:, 0:1]
            return np.real(-g)
        def func1_inner(x):
            normal = -self.inner.boundary_normal(x)
            g = 1j * self.k0 * np.exp(1j * self.k0 * x[:, 0:1]) * normal[:, 0:1]
            return np.imag(-g)
        def func0_outer(x, y):
            result = -self.k0 * y[:, 1:2]
            return result
        def func1_outer(x, y):
            result = self.k0 * y[:, 0:1]
            return result

        bc0_inner = dde.NeumannBC(self.geomtime, func0_inner, boundary_inner, component = 0)
        bc1_inner = dde.NeumannBC(self.geomtime, func1_inner, boundary_inner, component = 1)

        bc0_outer = dde.RobinBC(self.geomtime, func0_outer, boundary_outer, component = 0)
        bc1_outer = dde.RobinBC(self.geomtime, func1_outer, boundary_outer, component = 1)

        bcs = [bc0_inner, bc1_inner, bc0_outer, bc1_outer]
        
        return dde.data.PDE(self.geomtime, self.pde, bcs, num_domain=self.NumDomain, num_boundary=self.NumBoundary, solution=self.sol,num_test=self.NumTest)
    
    def set_axes(self, axes):
        axes.set_xlim(-self.length / 2, self.length / 2)
        axes.set_ylim(-self.length / 2, self.length / 2)
        axes.set_xlabel('x1')
        axes.set_ylabel('x2')

    def plot_data(self, X, axes=None):
        from matplotlib import pyplot as plt
        if axes is None:
            fig, axes = plt.subplots()
        self.set_axes(axes)
        axes.scatter(X[:, 0], X[:, 1])
        return axes
    
    def plot_heatmap_at_axes(self, X, y, axes, title):
        axes.set_title(title)
        self.set_axes(axes)
        return axes.pcolormesh(X[:, 0].reshape(1000, 1000), X[:, 1].reshape(1000, 1000), y.reshape(1000, 1000), cmap='rainbow')
    
    def plot_result(self, solver, colorbar=[0,0,0]):
        from matplotlib import pyplot as plt
        X = np.array([[x1, x2] for x1 in np.linspace(-self.length / 2, self.length / 2, 1000) for x2 in np.linspace(-self.length / 2, self.length / 2, 1000)])
        y = self.sol(X)[:,0]
        y[ self.geomtime.inside(X) == 0 ] = np.nan
        model_y = solver.model.predict(X)[:,0]
        model_y[self.geomtime.inside(X) == 0 ] = np.nan

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