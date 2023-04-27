import numpy as np
import deepxde as dde
from deepxde.backend import backend_name
from deepxde.backend import torch, paddle
from abc import ABC, abstractmethod

class PDECases(ABC):
    def __init__(self, 
                 name,
                 NumDomain=2000,
                 use_output_transform=False,
                 layer_size = [2] + [32] * 3 + [1],
                 activation = "tanh",
                 initializer = "Glorot uniform"):
        self.name = name
        self.NumDomain = NumDomain
        self.use_output_transform = use_output_transform
        self.backend = self.gen_backend()
        self.net = self.gen_net(layer_size, activation, initializer)
        self.pde = self.gen_pde()
        self.geomtime = self.gen_geomtime()
        self.data = self.gen_data()
        

    def gen_backend(self):
        if backend_name == 'pytorch':
            return torch
        elif backend_name == 'paddle':
            return paddle
        else:
            raise Warning('Currently only support pytorch and paddle backend')
           
    def gen_net(self, layer_size, activation, initializer):
        net = dde.maps.FNN(layer_size, activation, initializer)
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

class Burgers(PDECases):
    def __init__(self, 
                 NumDomain=2000, 
                 layer_size=[2] + [64] * 3 + [1], 
                 activation="tanh", 
                 initializer="Glorot normal"):
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
        return dde.data.TimePDE(self.geomtime, self.pde, [], num_domain=self.NumDomain, num_test=10000, train_distribution="pseudo")
    
    def gen_testdata(self):
        data = np.load("./data/Burgers.npz")
        t, x, exact = data["t"], data["x"], data["usol"].T
        xx, tt = np.meshgrid(x, t)
        X = np.vstack((np.ravel(xx), np.ravel(tt))).T
        y = exact.flatten()[:, None]
        return X, y
    
    def output_transform(self, x, y):
        return -self.backend.sin(np.pi * x[:, 0:1]) + (1 - x[:, 0:1] ** 2) * (x[:, 1:]) * y
    
    def plot_heatmap_at_axes(self, X, y, axes, title):
        axes.set_title(title)
        axes.set_xlabel("t")
        axes.set_ylabel("x")
        axes.pcolormesh(X[:, 1].reshape(100, 256), X[:, 0].reshape(100, 256), y.reshape(100, 256), cmap='rainbow')
    
    def plot_heatmap(self, solver):
        from matplotlib import pyplot as plt
        X, y = self.gen_testdata()
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        
        self.plot_heatmap_at_axes(X, y, axes=axes[0][0], title="Exact solution")
        model_y = solver.model.predict(X)
        self.plot_heatmap_at_axes(X, model_y, axes[1][0], title=solver.name)
        self.plot_heatmap_at_axes(X, np.abs(model_y - y) , axes[1][1], title="Absolute error")
        plt.show()
        return fig, axes
    

class AllenCahn(PDECases):
    def __init__(self, 
                 NumDomain=2000, 
                 layer_size=[2] + [64] * 3 + [1], 
                 activation="tanh", 
                 initializer="Glorot normal"):
        super().__init__(name='AllenCahn', NumDomain=NumDomain, use_output_transform=True, layer_size=layer_size, activation=activation, initializer=initializer)

    def gen_net(self):
        return dde.maps.FNN([2] + [64] * 3 + [1], "tanh", "Glorot normal")

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
        return dde.data.TimePDE(self.geomtime, self.pde, [], num_domain=self.NumDomain, num_test=10000, train_distribution="pseudo")

    def gen_testdata(self):
        from scipy.io import loadmat
        data = loadmat("./data/Allen_Cahn.mat")
        t = data["t"]
        x = data["x"]
        u = data["u"]
        dt = dx = 0.01
        xx, tt = np.meshgrid(x, t)
        X = np.vstack((np.ravel(xx), np.ravel(tt))).T
        y = u.flatten()[:, None]
        return X, y
    
    def output_transform(self, x, y):
        x_in = x[:, 0:1]
        t_in = x[:, 1:2]
        return t_in * (1 + x_in) * (1 - x_in) * y + self.backend.square(x_in) * self.backend.cos(np.pi * x_in)
    
class Diffusion(PDECases):
    def __init__(self, 
                 NumDomain=2000, 
                 layer_size=[2] + [32] * 3 + [1], 
                 activation="tanh", 
                 initializer="Glorot normal"):
        super().__init__(name='Diffusion', NumDomain=NumDomain, use_output_transform=True, layer_size=layer_size, activation=activation, initializer=initializer)

    def gen_net(self):
        return dde.maps.FNN([2] + [32] * 3 + [1], "tanh", "Glorot uniform")

    def gen_pde(self):
        def pde(x, y):
            dy_t = dde.grad.jacobian(y, x, j=1)
            dy_xx = dde.grad.hessian(y, x, j=0)
            return (
                    dy_t
                    - dy_xx
                    + self.backend.exp(-x[:, 1:])
                    * (self.backend.sin(np.pi * x[:, 0:1]) - np.pi ** 2 * self.backend.sin(np.pi * x[:, 0:1]))
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
        return self.backend.sin(np.pi * x[:, 0:1]) + (1 - x[:, 0:1] ** 2) * (x[:, 1:]) * y
        
