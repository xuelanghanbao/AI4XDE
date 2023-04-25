import os
import time
import numpy as np
import deepxde as dde
from deepxde.backend import torch

class PINNSolver():
    def __init__(self, name, case='Burgers', NumDomain=2000):
        self.name = name
        self.NumDomain = NumDomain
        self.geomtime = self.gen_geomtime()
        self.net = self.gen_net()

        def pde(x, y):
            dy_x = dde.grad.jacobian(y, x, i=0, j=0)
            dy_t = dde.grad.jacobian(y, x, i=0, j=1)
            dy_xx = dde.grad.hessian(y, x, i=0, j=0)
            return dy_t + y * dy_x - 0.01 / np.pi * dy_xx
        self.pde = pde
        self.data = self.gen_data()
        
        self.model = dde.Model(self.data, self.net)
        self.error = []
        self.losshistory = None
        self.train_state = None
        self.train_cost = None
    
    def gen_geomtime(self):
        geom = dde.geometry.Interval(-1, 1)
        timedomain = dde.geometry.TimeDomain(0, 1)
        return dde.geometry.GeometryXTime(geom, timedomain)
    
    def gen_net(self):
        net = dde.maps.FNN([2] + [64] * 3 + [1], "tanh", "Glorot normal")
        net.apply_output_transform(self.output_transform)
        return net
    
    def gen_data(self):
        return dde.data.TimePDE(self.geomtime, self.pde, [], num_domain=self.NumDomain, num_test=10000, train_distribution="pseudo")
    
    def output_transform(self, x, y):
        return -torch.sin(np.pi * x[:, 0:1]) + (1 - x[:, 0:1] ** 2) * (x[:, 1:]) * y
    
    def gen_testdata(self):
        data = np.load("../data/Burgers.npz")
        t, x, exact = data["t"], data["x"], data["usol"].T
        xx, tt = np.meshgrid(x, t)
        X = np.vstack((np.ravel(xx), np.ravel(tt))).T
        y = exact.flatten()[:, None]
        return X, y
    
    def save(self, add_time=False):
        if add_time:
            path = f'./models/{self.name}_{time.strftime("%Y%m%d_%H%M%S", time.localtime())}/'
        else:
            path = f'./models/{self.name}/'
        
        if not os.path.exists(path):
            os.makedirs(path)
        self.model.save(path)
        dde.saveplot(self.losshistory, self.train_state, issave=True, isplot=False, output_dir=path)
        np.savetxt(f'{path}{self.name}_error.txt', self.error)

    def eval(self):
        X, y = self.gen_testdata()
        y_pred = self.model.predict(X)
        error = dde.metrics.l2_relative_error(y, y_pred)
        print("L2 relative error:", error)
        self.error.append(np.array(error))

    def train_step(self, lr=1e-3, iteration=15000, callbacks=None, eval=True):
        self.model.compile("adam", lr=lr)
        self.model.train(iterations=iteration, callbacks=callbacks)
        self.model.compile("L-BFGS")
        self.losshistory, self.train_state = self.model.train()
        if eval:
            self.eval()

    def closure(self):
        self.train_step()
    
    def train(self):
        t_start = time.time()
        print(f"Model {self.name} is training...")

        self.closure()

        t_end = time.time()
        self.train_cost = t_end - t_start