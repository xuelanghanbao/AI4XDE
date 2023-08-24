import numpy as np
import deepxde as dde
import deepxde.backend as bkd
from .PDECases import PDECases
from abc import abstractmethod
from ..utils.Visualization import *


class ConvectionCase1D(PDECases):
    def __init__(
        self,
        name,
        beta,
        interval,
        time_interval,
        **kwargs,
    ):
        self.beta = beta
        self.interval = interval
        self.time_interval = time_interval
        super().__init__(
            name=name,
            visualization=Visualization_2D(
                x_limit=time_interval,
                y_limit=interval,
                x_label="t",
                y_label="x",
                feature_transform=lambda X: X[:, [1, 0]],
            ),
            **kwargs,
        )

    def gen_data(self):
        ic = dde.icbc.IC(
            self.geomtime,
            self.ic_func,
            lambda _, on_initial: on_initial,
        )
        return dde.data.TimePDE(
            self.geomtime,
            self.pde,
            ic,
            num_domain=self.NumDomain,
            num_boundary=80,
            solution=self.sol,
            num_test=2540,
            num_initial=160,
        )

    def gen_geomtime(self):
        geom = dde.geometry.Interval(self.interval[0], self.interval[1])
        timedomain = dde.geometry.TimeDomain(
            self.time_interval[0], self.time_interval[1]
        )
        return dde.geometry.GeometryXTime(geom, timedomain)

    def gen_pde(self):
        def pde(x, y):
            dy_t = dde.grad.jacobian(y, x, i=0, j=1)
            dy_xx = dde.grad.hessian(y, x, i=0, j=0)
            return dy_t - self.beta * dy_xx

        return pde

    def plot_result(self, solver, colorbar=[0, 0, 0]):
        X = np.array(
            [
                [x, t]
                for x in np.linspace(self.interval[0], self.interval[1], 1000)
                for t in np.linspace(self.time_interval[0], self.time_interval[1], 1000)
            ]
        )
        y = self.sol(X)
        model_y = solver.model.predict(X)

        fig, axes = self.Visualization.plot_exact_predict_error_2D(
            X, y, model_y, shape=[1000, 1000], title=solver.name, colorbar=colorbar
        )
        return fig, axes


class Convection_1D_Periodic(ConvectionCase1D):
    """Case of 1D Convection with Periodic
    Implementation of this case in paper http://arxiv.org/abs/2109.01050.
    The code of this paper is available at https://github.com/a1k12/characterizing-pinns-failure-modes.
    """

    def __init__(
        self,
        beta=30,
        interval=[0, 2 * np.pi],
        time_end=1,
        N=256,
        nt=100,
        NumDomain=1000,
        ic_func=lambda x: np.sin(x[0]),
        layer_size=[2] + [32] * 3 + [1],
        **kwargs,
    ):
        self.ic_func = ic_func
        self.N = N
        self.nt = nt
        super().__init__(
            name="1D Convection with Periodic",
            beta=beta,
            interval=interval,
            time_interval=[0, time_end],
            NumDomain=NumDomain,
            layer_size=layer_size,
            **kwargs,
        )

    def gen_testdata(self, source=0):
        h = (self.interval[1] - self.interval[0]) / self.N
        x = np.arange(
            self.interval[0], self.interval[1], h
        )  # not inclusive of the last point
        t = np.linspace(self.time_interval[0], self.time_interval[1], self.nt).reshape(
            -1, 1
        )
        X, T = np.meshgrid(x, t)

        u0 = [self.ic_func([x_i]) for x_i in x]
        G = (np.copy(u0) * 0) + source  # G is the same size as u0

        IKX_pos = 1j * np.arange(0, self.N / 2 + 1, 1)
        IKX_neg = 1j * np.arange(-self.N / 2 + 1, 0, 1)
        IKX = np.concatenate((IKX_pos, IKX_neg))

        uhat0 = np.fft.fft(u0)
        nu_factor = np.exp(-self.beta * IKX * T)
        A = uhat0 - np.fft.fft(G) * 0  # at t=0, second term goes away
        uhat = A * nu_factor + np.fft.fft(G) * T  # for constant, fft(p) dt = fft(p)*T
        u = np.real(np.fft.ifft(uhat))

        u_vals = u.flatten()
        x_vals = X.flatten()
        t_vals = T.flatten()
        return np.array([x_vals, t_vals]).T, u_vals.reshape(-1, 1)

    def gen_data(self):
        ic = dde.icbc.IC(
            self.geomtime,
            self.ic_func,
            lambda _, on_initial: on_initial,
        )
        bc = dde.icbc.PeriodicBC(
            self.geomtime,
            0,
            lambda x, on_boundary: on_boundary and np.isclose(x[0], 2 * np.pi),
        )
        return dde.data.TimePDE(
            self.geomtime,
            self.pde,
            [bc, ic],
            num_domain=self.NumDomain,
            num_boundary=80,
            num_test=2540,
            num_initial=160,
        )

    def plot_result(self, solver, colorbar=[0, 0, 0]):
        X, y = self.get_testdata()
        model_y = solver.model.predict(X)
        print(f"{y.shape=}, {model_y.shape=}")

        fig, axes = self.Visualization.plot_exact_predict_error_2D(
            X, y, model_y, shape=[100, 256], title=solver.name, colorbar=colorbar
        )
        return fig, axes
