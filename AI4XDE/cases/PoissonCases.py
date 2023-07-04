import numpy as np
import deepxde as dde
import deepxde.backend as bkd
from .PDECases import PDECases
from abc import abstractmethod

class PoissonCase1D(PDECases):
    def __init__(self, 
                 name,
                 NumDomain=2000,
                 Interval=None,
                 use_output_transform=False,
                 inverse=False,
                 layer_size = [2] + [32] * 3 + [1],
                 activation = 'tanh',
                 initializer = 'Glorot uniform'):
        self.Interval = Interval
        super().__init__(name=name, NumDomain=NumDomain, use_output_transform=use_output_transform, inverse=inverse, layer_size=layer_size, activation=activation, initializer=initializer)

    @abstractmethod
    def func(self,x):
        pass

    @abstractmethod
    def gen_data(self):
        pass

    def gen_geomtime(self):
        return dde.geometry.Interval(self.Interval[0], self.Interval[1])

    def gen_pde(self):
        def pde(x, y):
            dy_xx = dde.grad.hessian(y, x)
            return dy_xx - self.func(x)
        return pde
    
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
    
class Poisson_1D_Dirichlet(PoissonCase1D):
    def __init__(self, 
                 NumDomain=16, 
                 layer_size=[1] + [50] * 3 + [1], 
                 activation='tanh', 
                 initializer='Glorot uniform'):
        self.Interval = [-1, 1]
        super().__init__(name='Poisson equation in 1D with Dirichlet boundary conditions', NumDomain=NumDomain, Interval=self.Interval , use_output_transform=False, layer_size=layer_size, activation=activation, initializer=initializer)

    def func(self,x):
        return - np.pi**2 * bkd.sin(np.pi * x)

    def sol(self,x):
        return np.sin(np.pi * x)
    
    def gen_data(self):
        bc = dde.icbc.DirichletBC(self.geomtime, self.sol, lambda _, on_boundary: on_boundary)
        return dde.data.PDE(self.geomtime, self.pde, bc, self.NumDomain, 2, solution=self.sol, num_test=100)
    
class Poisson_1D_Dirichlet_Neumann(PoissonCase1D):
    def __init__(self, 
                 NumDomain=16, 
                 layer_size=[1] + [50] * 3 + [1], 
                 activation='tanh', 
                 initializer='Glorot uniform'):
        self.Interval = [-1, 1]
        super().__init__(name='Poisson equation in 1D with Dirichlet/Neumann boundary conditions', NumDomain=NumDomain, Interval=self.Interval , use_output_transform=False, layer_size=layer_size, activation=activation, initializer=initializer)

    def func(self,x):
        return 2

    def sol(self,x):
        return (x+1)**2
    
    def gen_data(self):
        def boundary_l(x, on_boundary):
            return on_boundary and np.isclose(x[0], -1)
        def boundary_r(x, on_boundary):
            return on_boundary and np.isclose(x[0], 1)
        bc_l = dde.icbc.DirichletBC(self.geomtime, self.sol, boundary_l)
        bc_r = dde.icbc.NeumannBC(self.geomtime, lambda X: 2 * (X + 1), boundary_r)
        return dde.data.PDE(self.geomtime, self.pde, [bc_l, bc_r], self.NumDomain, 2, solution=self.sol, num_test=100)
    
class Poisson_1D_Dirichlet_Robin(PoissonCase1D):
    def __init__(self, 
                 NumDomain=16, 
                 layer_size=[1] + [50] * 3 + [1], 
                 activation='tanh', 
                 initializer='Glorot uniform'):
        self.Interval = [-1, 1]
        super().__init__(name='Poisson equation in 1D with Dirichlet/Robin boundary conditions', NumDomain=NumDomain, Interval=self.Interval , use_output_transform=False, layer_size=layer_size, activation=activation, initializer=initializer)

    def func(self,x):
        return 2

    def sol(self,x):
        return (x+1)**2
    
    def gen_data(self):
        def boundary_l(x, on_boundary):
            return on_boundary and np.isclose(x[0], -1)
        def boundary_r(x, on_boundary):
            return on_boundary and np.isclose(x[0], 1)
        bc_l = dde.icbc.DirichletBC(self.geomtime, self.sol, boundary_l)
        bc_r = dde.icbc.RobinBC(self.geomtime, lambda X, y: y, boundary_r)
        return dde.data.PDE(self.geomtime, self.pde, [bc_l, bc_r], self.NumDomain, 2, solution=self.sol, num_test=100)
    
class Poisson_1D_Dirichlet_Periodic(PoissonCase1D):
    def __init__(self, 
                 NumDomain=16, 
                 layer_size=[1] + [50] * 3 + [1], 
                 activation='tanh', 
                 initializer='Glorot uniform'):
        self.Interval = [-1, 1]
        super().__init__(name='Poisson equation in 1D with Dirichlet/Periodic boundary conditions', NumDomain=NumDomain, Interval=self.Interval , use_output_transform=False, layer_size=layer_size, activation=activation, initializer=initializer)

    def func(self,x):
        return - np.pi**2 * bkd.sin(np.pi * x)

    def sol(self,x):
        return np.sin(np.pi * x)
    
    def gen_data(self):
        def boundary_l(x, on_boundary):
            return on_boundary and np.isclose(x[0], -1)
        def boundary_r(x, on_boundary):
            return on_boundary and np.isclose(x[0], 1)
        bc_l = dde.icbc.DirichletBC(self.geomtime, self.sol, boundary_l)
        bc2 = dde.icbc.PeriodicBC(self.geomtime, 0, boundary_r)
        return dde.data.PDE(self.geomtime, self.pde, [bc_l, bc2], self.NumDomain, 2, solution=self.sol, num_test=100)
    
class Poisson_1D_Dirichlet_PointSetOperator(PoissonCase1D):
    def __init__(self, 
                 NumDomain=16, 
                 layer_size=[1] + [50] * 3 + [1], 
                 activation='tanh', 
                 initializer='Glorot uniform'):
        self.Interval = [-1, 1]
        super().__init__(name='Poisson equation in 1D with Dirichlet/PointSetOperator boundary conditions', NumDomain=NumDomain, Interval=self.Interval , use_output_transform=False, layer_size=layer_size, activation=activation, initializer=initializer)

    def func(self,x):
        return 2

    def sol(self,x):
        return (x+1)**2
    
    def gen_data(self):
        def boundary_l(x, on_boundary):
            return on_boundary and np.isclose(x[0], -1)
        def dy_x(x, y, X):
            dy_x = dde.grad.jacobian(y, x)
            return dy_x
        def d_func(x):
            return 2 * (x + 1)
        bc_l = dde.icbc.DirichletBC(self.geomtime, self.sol, boundary_l)
        boundary_pts = self.geomtime.random_boundary_points(2)
        r_boundary_pts = boundary_pts[np.isclose(boundary_pts, 1)].reshape(-1, 1)
        bc_r = dde.icbc.PointSetOperatorBC(r_boundary_pts, d_func(r_boundary_pts), dy_x)
        return dde.data.PDE(self.geomtime, self.pde, [bc_l, bc_r], self.NumDomain, 2, solution=self.sol, num_test=100)
    
class Poisson_1D_Hard_Boundary(PoissonCase1D):
    def __init__(self, 
                 NumDomain=64, 
                 layer_size=[1] + [50] * 3 + [1], 
                 activation='tanh', 
                 initializer='Glorot uniform'):
        self.Interval = [0, np.pi]
        super().__init__(name='Poisson equation in 1D with hard boundary conditions', NumDomain=NumDomain, Interval=self.Interval , use_output_transform=True, layer_size=layer_size, activation=activation, initializer=initializer)

    def func(self,x):
        summation = sum([i * bkd.sin(i * x) for i in range(1, 5)])
        return -summation - 8 * bkd.sin(8 * x)

    def sol(self,x):
        summation = sum([np.sin(i * x) / i for i in range(1, 5)])
        return x + summation + np.sin(8 * x) / 8
    
    def gen_data(self):
        return dde.data.PDE(self.geomtime, self.pde, [], self.NumDomain, solution=self.sol, num_test=400)
    
    def output_transform(self, x, y):
        return x * (np.pi - x) * y + x
    
class Poisson_1D_Fourier_Net(PoissonCase1D):
    def __init__(self, 
                 NumDomain=1280, 
                 layer_size=[1] + [100] * 3 + [1], 
                 activation='tanh', 
                 initializer='Glorot uniform'):
        self.Interval = [0, 1]
        self.A = 2
        self.B = 50
        super().__init__(name='Poisson equation in 1D with hard boundary conditions', NumDomain=NumDomain, Interval=self.Interval , use_output_transform=True, layer_size=layer_size, activation=activation, initializer=initializer)

    def func(self,x):
        result = -(np.pi * self.A) ** 2 * bkd.sin(np.pi * self.A * x) - 0.1 * (np.pi * self.B) ** 2 * bkd.sin(np.pi * self.B * x)
        return result

    def sol(self,x):
        return np.sin(np.pi * self.A * x) + 0.1 * np.sin(np.pi * self.B * x)
    
    def gen_net(self, layer_size, activation, initializer):
        return dde.nn.MsFFN(layer_size, activation, initializer, sigmas=[1, 10])

    def gen_data(self):
        bc = dde.icbc.DirichletBC(self.geomtime, self.sol, lambda _, on_boundary: on_boundary)
        return dde.data.PDE(self.geomtime, self.pde, bc, self.NumDomain, train_distribution="pseudo", solution=self.sol, num_test=10000)
    
class Poisson_2D_L_Shaped(PDECases):
    def __init__(self, 
                 NumDomain=1200, 
                 layer_size=[2] + [50] * 4 + [1], 
                 activation='tanh', 
                 initializer='Glorot uniform'):
        super().__init__(name='Poisson equation in 1D with hard boundary conditions', NumDomain=NumDomain, use_output_transform=False, layer_size=layer_size, activation=activation, initializer=initializer)

    def gen_data(self):
        def boundary(_, on_boundary):
            return on_boundary
        bc = dde.icbc.DirichletBC(self.geomtime, lambda x: 0, boundary)
        return dde.data.PDE(self.geomtime, self.pde, bc, num_domain=self.NumDomain, num_boundary=120, num_test=1500)
    
    def gen_geomtime(self):
        return dde.geometry.Polygon([[0, 0], [1, 0], [1, -1], [-1, -1], [-1, 1], [0, 1]])

    def gen_pde(self):
        def pde(x, y):
            dy_xx = dde.grad.hessian(y, x, i=0, j=0)
            dy_yy = dde.grad.hessian(y, x, i=1, j=1)
            return -dy_xx - dy_yy - 1
        return pde
    
    def gen_testdata(self):
        X = np.array([[x, t] for x in np.linspace(-1, 1, 1000) for t in np.linspace(-1, 1, 1000)])
        y = np.linspace(-1, 1, 1000).T
        return X, y
    
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
        axes.scatter(X[:, 0], X[:, 1])
        return axes
    
    def plot_heatmap_at_axes(self, X, y, axes, title):
        axes.set_title(title)
        self.set_axes(axes)
        return axes.pcolormesh(X[:, 0].reshape(1000, 1000), X[:, 1].reshape(1000, 1000), y.reshape(1000, 1000), cmap='rainbow')
    
    def plot_result(self, solver, colorbar=None):
        from matplotlib import pyplot as plt
        X = np.array([[x, t] for x in np.linspace(-1, 1, 1000) for t in np.linspace(-1, 1, 1000)])
        #y = self.sol(X)
        model_y = solver.model.predict(X)
        model_y[self.geomtime.inside(X) == 0] = np.nan

        fig, axes = plt.subplots()
        self.plot_heatmap_at_axes(X, model_y, axes, title=solver.name)
        plt.show()
        return fig, axes