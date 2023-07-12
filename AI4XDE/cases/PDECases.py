import numpy as np
import deepxde as dde
import deepxde.backend as bkd
from abc import ABC, abstractmethod


class PDECases(ABC):
    def __init__(
        self,
        name,
        NumDomain=2000,
        use_output_transform=False,
        inverse=False,
        layer_size=[2] + [32] * 3 + [1],
        activation="tanh",
        initializer="Glorot uniform",
    ):
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
            raise Warning("You must rewrite one of sol() and gen_testdata()")

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
    def __init__(
        self,
        NumDomain=2000,
        layer_size=[2] + [64] * 3 + [1],
        activation="tanh",
        initializer="Glorot normal",
    ):
        super().__init__(
            name="Burgers",
            NumDomain=NumDomain,
            use_output_transform=True,
            layer_size=layer_size,
            activation=activation,
            initializer=initializer,
        )

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
        return dde.data.TimePDE(
            self.geomtime,
            self.pde,
            [],
            num_domain=self.NumDomain,
            num_test=10000,
            train_distribution="pseudo",
        )

    def gen_testdata(self):
        import os

        basepath = os.path.abspath(__file__)
        folder = os.path.dirname(os.path.dirname(basepath))
        data_path = os.path.join(folder, "data/Burgers.npz")
        data = np.load(data_path)
        t, x, exact = data["t"], data["x"], data["usol"].T
        xx, tt = np.meshgrid(x, t)
        X = np.vstack((np.ravel(xx), np.ravel(tt))).T
        y = exact.flatten()[:, None]
        return X, y

    def output_transform(self, x, y):
        return -bkd.sin(np.pi * x[:, 0:1]) + (1 - x[:, 0:1] ** 2) * (x[:, 1:]) * y

    def set_axes(self, axes):
        axes.set_xlim(0, 1)
        axes.set_ylim(-1, 1)
        axes.set_xlabel("t")
        axes.set_ylabel("x")

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
        return axes.pcolormesh(
            X[:, 1].reshape(100, 256),
            X[:, 0].reshape(100, 256),
            y.reshape(100, 256),
            cmap="rainbow",
        )

    def plot_result(self, solver, colorbar=[0, 0, 0]):
        from matplotlib import pyplot as plt

        X, y = self.gen_testdata()
        model_y = solver.model.predict(X)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axs = []
        axs.append(
            self.plot_heatmap_at_axes(X, y, axes=axes[0], title="Exact solution")
        )
        axs.append(self.plot_heatmap_at_axes(X, model_y, axes[1], title=solver.name))
        axs.append(
            self.plot_heatmap_at_axes(
                X, np.abs(model_y - y), axes[2], title="Absolute error"
            )
        )

        for needColorbar, ax, axe in zip(colorbar, axs, axes):
            if needColorbar:
                fig.colorbar(ax, ax=axe)
        plt.show()
        return fig, axes


class AllenCahn(PDECases):
    """Case of Allen-Cahn equation.
    Implementation of Allen-Cahn equation example in paper https://arxiv.org/abs/2111.02801.
    """

    def __init__(
        self,
        NumDomain=2000,
        layer_size=[2] + [64] * 3 + [1],
        activation="tanh",
        initializer="Glorot normal",
    ):
        super().__init__(
            name="AllenCahn",
            NumDomain=NumDomain,
            use_output_transform=True,
            layer_size=layer_size,
            activation=activation,
            initializer=initializer,
        )

    def gen_pde(self):
        def pde(x, y):
            u = y
            du_xx = dde.grad.hessian(y, x, i=0, j=0)
            du_t = dde.grad.jacobian(y, x, j=1)
            return du_t - 0.001 * du_xx + 5 * (u**3 - u)

        return pde

    def gen_geomtime(self):
        geom = dde.geometry.Interval(-1, 1)
        timedomain = dde.geometry.TimeDomain(0, 1)
        return dde.geometry.GeometryXTime(geom, timedomain)

    def gen_data(self):
        return dde.data.TimePDE(
            self.geomtime,
            self.pde,
            [],
            num_domain=self.NumDomain,
            num_test=10000,
            train_distribution="pseudo",
        )

    def gen_testdata(self):
        import os
        from scipy.io import loadmat

        basepath = os.path.abspath(__file__)
        folder = os.path.dirname(os.path.dirname(basepath))
        data_path = os.path.join(folder, "data/Allen_Cahn.mat")
        data = loadmat(data_path)
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
        return t_in * (1 + x_in) * (1 - x_in) * y + bkd.square(x_in) * bkd.cos(
            np.pi * x_in
        )

    def set_axes(self, axes):
        axes.set_xlim(0, 1)
        axes.set_ylim(-1, 1)
        axes.set_xlabel("t")
        axes.set_ylabel("x")

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
        return axes.pcolormesh(
            X[:, 1].reshape(101, 201),
            X[:, 0].reshape(101, 201),
            y.reshape(101, 201),
            cmap="rainbow",
        )

    def plot_result(self, solver, colorbar=[0, 0, 0]):
        from matplotlib import pyplot as plt

        X, y = self.gen_testdata()
        model_y = solver.model.predict(X)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axs = []
        axs.append(
            self.plot_heatmap_at_axes(X, y, axes=axes[0], title="Exact solution")
        )
        axs.append(self.plot_heatmap_at_axes(X, model_y, axes[1], title=solver.name))
        axs.append(
            self.plot_heatmap_at_axes(
                X, np.abs(model_y - y), axes[2], title="Absolute error"
            )
        )

        for needColorbar, ax, axe in zip(colorbar, axs, axes):
            if needColorbar:
                fig.colorbar(ax, ax=axe)
        plt.show()
        return fig, axes


class Diffusion(PDECases):
    def __init__(
        self,
        NumDomain=40,
        use_output_transform=True,
        layer_size=[2] + [32] * 3 + [1],
        activation="tanh",
        initializer="Glorot normal",
    ):
        super().__init__(
            name="Diffusion",
            NumDomain=NumDomain,
            use_output_transform=use_output_transform,
            layer_size=layer_size,
            activation=activation,
            initializer=initializer,
        )

    def gen_pde(self):
        def pde(x, y):
            dy_t = dde.grad.jacobian(y, x, j=1)
            dy_xx = dde.grad.hessian(y, x, j=0)
            return (
                dy_t
                - dy_xx
                + bkd.exp(-x[:, 1:])
                * (bkd.sin(np.pi * x[:, 0:1]) - np.pi**2 * bkd.sin(np.pi * x[:, 0:1]))
            )

        return pde

    def gen_geomtime(self):
        geom = dde.geometry.Interval(-1, 1)
        timedomain = dde.geometry.TimeDomain(0, 1)
        return dde.geometry.GeometryXTime(geom, timedomain)

    def sol(self, x):
        return np.sin(np.pi * x[:, 0:1]) * np.exp(-x[:, 1:])

    def gen_data(self):
        if self.use_output_transform:
            data = dde.data.TimePDE(
                self.geomtime,
                self.pde,
                [],
                num_domain=self.NumDomain,
                solution=self.sol,
                num_test=10000,
            )
        else:
            bc = dde.icbc.DirichletBC(
                self.geomtime, self.sol, lambda _, on_boundary: on_boundary
            )
            ic = dde.icbc.IC(self.geomtime, self.sol, lambda _, on_initial: on_initial)
            icbc = [bc, ic]
            data = dde.data.TimePDE(
                self.geomtime,
                self.pde,
                icbc,
                num_domain=self.NumDomain,
                num_boundary=20,
                num_initial=10,
                solution=self.sol,
                num_test=10000,
            )
        return data

    def output_transform(self, x, y):
        return bkd.sin(np.pi * x[:, 0:1]) + (1 - x[:, 0:1] ** 2) * (x[:, 1:]) * y

    def set_axes(self, axes):
        axes.set_xlim(0, 1)
        axes.set_ylim(-1, 1)
        axes.set_xlabel("t")
        axes.set_ylabel("x")

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
        return axes.pcolormesh(
            X[:, 1].reshape(1000, 1000),
            X[:, 0].reshape(1000, 1000),
            y.reshape(1000, 1000),
            cmap="rainbow",
        )

    def plot_result(self, solver, colorbar=[0, 0, 0]):
        from matplotlib import pyplot as plt

        X = np.array(
            [[x, t] for x in np.linspace(-1, 1, 1000) for t in np.linspace(0, 1, 1000)]
        )
        y = self.sol(X)
        model_y = solver.model.predict(X)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axs = []
        axs.append(
            self.plot_heatmap_at_axes(X, y, axes=axes[0], title="Exact solution")
        )
        axs.append(self.plot_heatmap_at_axes(X, model_y, axes[1], title=solver.name))
        axs.append(
            self.plot_heatmap_at_axes(
                X, np.abs(model_y - y), axes[2], title="Absolute error"
            )
        )

        for needColorbar, ax, axe in zip(colorbar, axs, axes):
            if needColorbar:
                fig.colorbar(ax, ax=axe)
        plt.show()
        return fig, axes


class Diffusion_Reaction_Inverse(PDECases):
    def __init__(
        self,
        NumDomain=2000,
        use_output_transform=False,
        layer_size=[1, [20, 20], [20, 20], [20, 20], 2],
        activation="tanh",
        initializer="Glorot uniform",
    ):
        self.res = self.gen_res()
        self.metrics = self.gen_metrics()
        super().__init__(
            name="Diffusion_Reaction_Inverse",
            NumDomain=NumDomain,
            use_output_transform=use_output_transform,
            inverse=True,
            layer_size=layer_size,
            activation=activation,
            initializer=initializer,
        )

    def sol(self, x):
        return self.res.sol(x)[0]

    def gen_res(self):
        from scipy.integrate import solve_bvp

        def k(x):
            return 0.1 + np.exp(-0.5 * (x - 0.5) ** 2 / 0.15**2)

        def fun(x, y):
            return np.vstack((y[1], 100 * (k(x) * y[0] + np.sin(2 * np.pi * x))))

        def bc(ya, yb):
            return np.array([ya[0], yb[0]])

        a = np.linspace(0, 1, 1000)
        b = np.zeros((2, a.size))

        res = solve_bvp(fun, bc, a, b)
        return res

    def get_metrics(self, model):
        xx = np.linspace(0, 1, 1001)[:, None]

        def k(x):
            return 0.1 + np.exp(-0.5 * (x - 0.5) ** 2 / 0.15**2)

        def l2_u(_, __):
            return dde.metrics.l2_relative_error(
                self.sol(xx), model.predict(xx)[:, 0:1]
            )

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
        bc = dde.DirichletBC(
            self.geomtime, self.sol, lambda _, on_boundary: on_boundary, component=0
        )
        return dde.data.PDE(
            self.geomtime,
            self.pde,
            bcs=[bc, observe_u],
            num_domain=self.NumDomain - 2,
            num_boundary=2,
            train_distribution="pseudo",
            num_test=1000,
        )


class A_Simple_ODE(PDECases):
    def __init__(
        self,
        NumDomain=2000,
        layer_size=[1] + [64] * 3 + [2],
        activation="tanh",
        initializer="Glorot normal",
    ):
        super().__init__(
            name="A Simple ODE",
            NumDomain=NumDomain,
            use_output_transform=False,
            layer_size=layer_size,
            activation=activation,
            initializer=initializer,
        )

    def gen_pde(self):
        def ode(x, y):
            y1, y2 = y[:, 0:1], y[:, 1:]
            dy1_x = dde.grad.jacobian(y, x, i=0)
            dy2_x = dde.grad.jacobian(y, x, i=1)
            return [dy1_x - y2, dy2_x + y1]

        return ode

    def gen_geomtime(self):
        geom = dde.geometry.TimeDomain(0, 10 * np.pi)
        return geom

    def gen_data(self):
        def boundary(_, on_initial):
            return on_initial

        ic1 = dde.icbc.IC(self.geomtime, lambda x: 0, boundary, component=0)
        ic2 = dde.icbc.IC(self.geomtime, lambda x: 1, boundary, component=1)
        return dde.data.PDE(
            self.geomtime,
            self.pde,
            [ic1, ic2],
            num_domain=self.NumDomain,
            num_boundary=2,
            solution=self.sol,
            num_test=10000,
            train_distribution="pseudo",
        )

    def sol(self, x):
        return np.hstack((np.sin(x), np.cos(x)))

    def plot_result(self, model, axes=None, exact=True):
        from matplotlib import pyplot as plt

        xx = np.linspace(-np.pi / 2, np.pi / 2, 1001)[:, None]
        if axes is None:
            fig, axes = plt.subplots()
        if exact:
            axes.plot(xx, self.sol(xx), label="Exact")
        axes.plot(xx, model.predict(xx), "--", label="Prediction")
        axes.legend()
        axes.set_xlabel("x")
        axes.set_ylabel("y")
        axes.set_title(self.name)
        return fig, axes


class LotkaVolterra(PDECases):
    def __init__(
        self,
        NumDomain=3000,
        layer_size=[1] + [64] * 6 + [2],
        activation="tanh",
        initializer="Glorot normal",
    ):
        super().__init__(
            name="Lotka-Volterra",
            NumDomain=NumDomain,
            use_output_transform=False,
            layer_size=layer_size,
            activation=activation,
            initializer=initializer,
        )
        self.ub = 200
        self.rb = 20

    def gen_pde(self):
        def ode_system(x, y):
            r = y[:, 0:1]
            p = y[:, 1:2]
            dr_t = dde.grad.jacobian(y, x, i=0)
            dp_t = dde.grad.jacobian(y, x, i=1)
            return [
                dr_t
                - 1
                / self.ub
                * self.rb
                * (2.0 * self.ub * r - 0.04 * self.ub * r * self.ub * p),
                dp_t
                - 1
                / self.ub
                * self.rb
                * (0.02 * r * self.ub * p * self.ub - 1.06 * p * self.ub),
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
        return bkd.concat(
            [y1 * bkd.tanh(t) + 100 / self.ub, y2 * bkd.tanh(t) + 15 / self.ub], axis=1
        )

    def input_transform(self, t):
        return bkd.concat(
            (bkd.sin(t),),
            axis=1,
        )

    def gen_geomtime(self):
        geom = dde.geometry.TimeDomain(0, 1.0)
        return geom

    def gen_data(self):
        return dde.data.PDE(
            self.geomtime,
            self.pde,
            [],
            num_domain=self.NumDomain,
            num_boundary=2,
            num_test=3000,
        )

    def gen_testdata(self):
        from scipy import integrate

        def sol(t, r):
            x, y = r
            dx_t = (
                1
                / self.ub
                * self.rb
                * (2.0 * self.ub * x - 0.04 * self.ub * x * self.ub * y)
            )
            dy_t = (
                1
                / self.ub
                * self.rb
                * (0.02 * self.ub * x * self.ub * y - 1.06 * self.ub * y)
            )
            return dx_t, dy_t

        t = np.linspace(0, 1, 100)

        sol = integrate.solve_ivp(sol, (0, 10), (100 / self.ub, 15 / self.ub), t_eval=t)
        x_true, y_true = sol.y
        x_true = x_true.reshape(100, 1)
        y_true = y_true.reshape(100, 1)

        return x_true, y_true

    def plot_result(self, solver, axes=None, exact=True):
        from matplotlib import pyplot as plt

        X, y = self.gen_testdata()
        if axes is None:
            fig, axes = plt.subplots()
        if exact:
            axes.plot(X, y, label="Exact")
        axes.plot(X, solver.model.predict(X), "--", label="Prediction")
        axes.legend()
        axes.set_xlabel("t")
        axes.set_ylabel("population")
        axes.set_title(self.name)
        return fig, axes


class SecondOrderODE(PDECases):
    def __init__(
        self,
        NumDomain=16,
        layer_size=[1] + [50] * 3 + [1],
        activation="tanh",
        initializer="Glorot normal",
    ):
        super().__init__(
            name="Second Order ODE",
            NumDomain=NumDomain,
            use_output_transform=False,
            layer_size=layer_size,
            activation=activation,
            initializer=initializer,
        )
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
        return dde.data.TimePDE(
            self.geomtime,
            self.pde,
            [ic1, ic2],
            self.NumDomain,
            2,
            solution=self.sol,
            num_test=500,
        )

    def plot_result(self, solver, axes=None, exact=True):
        from matplotlib import pyplot as plt

        X, y = self.gen_testdata()
        if axes is None:
            fig, axes = plt.subplots()
        if exact:
            axes.plot(X, y, label="Exact")
        axes.plot(X, solver.model.predict(X), "--", label="Prediction")
        axes.legend()
        axes.set_xlabel("t")
        axes.set_ylabel("y")
        axes.set_title(self.name)
        return fig, axes


class Laplace_disk(PDECases):
    def __init__(
        self,
        NumDomain=2540,
        layer_size=[2] + [20] * 3 + [1],
        activation="tanh",
        initializer="Glorot normal",
    ):
        super().__init__(
            name="Laplace equation on a disk",
            NumDomain=NumDomain,
            use_output_transform=False,
            layer_size=layer_size,
            activation=activation,
            initializer=initializer,
        )

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
        bc_rad = dde.icbc.DirichletBC(
            self.geomtime,
            lambda x: np.cos(x[:, 1:2]),
            lambda x, on_boundary: on_boundary and np.isclose(x[0], 1),
        )
        return dde.data.TimePDE(
            self.geomtime, self.pde, bc_rad, self.NumDomain, 80, solution=self.sol
        )

    def gen_net(self, layer_size, activation, initializer):
        def feature_transform(x):
            return bkd.concat(
                [x[:, 0:1] * bkd.sin(x[:, 1:2]), x[:, 0:1] * bkd.cos(x[:, 1:2])], axis=1
            )

        net = dde.nn.FNN(layer_size, activation, initializer)
        net.apply_feature_transform(feature_transform)
        return net

    def set_axes(self, axes):
        axes.set_xlim(-1, 1)
        axes.set_ylim(-1, 1)
        axes.set_xlabel("x1")
        axes.set_ylabel("x2")

    def plot_data(self, X, axes=None):
        from matplotlib import pyplot as plt

        if axes is None:
            fig, axes = plt.subplots()
        self.set_axes(axes)
        X = np.array([[x[0] * np.cos(x[1]), x[0] * np.sin(x[1])] for x in X])
        axes.scatter(X[:, 0], X[:, 1])
        return axes

    def plot_heatmap_at_axes(self, X, y, axes, title):
        axes.set_title(title)
        self.set_axes(axes)
        return axes.pcolormesh(
            X[:, 0].reshape(1000, 1000),
            X[:, 1].reshape(1000, 1000),
            y.reshape(1000, 1000),
            cmap="rainbow",
        )

    def plot_result(self, solver, colorbar=[0, 0, 0]):
        from matplotlib import pyplot as plt

        X = np.array(
            [
                [x1, x2]
                for x1 in np.linspace(0, 1, 1000)
                for x2 in np.linspace(0, 2 * np.pi, 1000)
            ]
        )
        X_R = np.array([[x[0] * np.cos(x[1]), x[0] * np.sin(x[1])] for x in X])
        y = self.sol(X)
        model_y = solver.model.predict(X)

        y[self.geomtime.inside(X) == 0] = np.nan
        model_y[self.geomtime.inside(X) == 0] = np.nan

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axs = []
        axs.append(
            self.plot_heatmap_at_axes(X_R, y, axes=axes[0], title="Exact solution")
        )
        axs.append(self.plot_heatmap_at_axes(X_R, model_y, axes[1], title=solver.name))
        axs.append(
            self.plot_heatmap_at_axes(
                X_R, np.abs(model_y - y), axes[2], title="Absolute error"
            )
        )

        for needColorbar, ax, axe in zip(colorbar, axs, axes):
            if needColorbar:
                fig.colorbar(ax, ax=axe)

        plt.show()
        return fig, axes


class Helmholtz(PDECases):
    def __init__(
        self,
        NumDomain=2540,
        hard_constraint=False,
        layer_size=[2] + [150] * 3 + [1],
        activation="sin",
        initializer="Glorot uniform",
    ):
        self.n = 2
        self.k0 = 2 * np.pi * self.n
        self.hard_constraint = hard_constraint
        super().__init__(
            name="Helmholtz equation over a 2D square domain",
            NumDomain=NumDomain,
            use_output_transform=False,
            layer_size=layer_size,
            activation=activation,
            initializer=initializer,
        )

    def gen_pde(self):
        def pde(x, y):
            dy_xx = dde.grad.hessian(y, x, i=0, j=0)
            dy_yy = dde.grad.hessian(y, x, i=1, j=1)

            f = (
                self.k0**2
                * bkd.sin(self.k0 * x[:, 0:1])
                * bkd.sin(self.k0 * x[:, 1:2])
            )
            return -dy_xx - dy_yy - self.k0**2 * y - f

        return pde

    def sol(self, x):
        return np.sin(self.k0 * x[:, 0:1]) * np.sin(self.k0 * x[:, 1:2])

    def gen_geomtime(self):
        return dde.geometry.Rectangle([0, 0], [1, 1])

    def gen_data(self):
        if self.hard_constraint == True:
            bc = []
        else:
            bc = dde.icbc.DirichletBC(
                self.geomtime, lambda x: 0, lambda _, on_boundary: on_boundary
            )

        precision_train = 10
        precision_test = 30
        wave_len = 1 / self.n

        hx_train = wave_len / precision_train
        nx_train = int(1 / hx_train)

        hx_test = wave_len / precision_test
        nx_test = int(1 / hx_test)
        return dde.data.PDE(
            self.geomtime,
            self.pde,
            bc,
            num_domain=nx_train**2,
            num_boundary=4 * nx_train,
            solution=self.sol,
            num_test=nx_test**2,
        )

    def set_axes(self, axes):
        axes.set_xlim(0, 1)
        axes.set_ylim(0, 1)
        axes.set_xlabel("x1")
        axes.set_ylabel("x2")

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
        return axes.pcolormesh(
            X[:, 0].reshape(1000, 1000),
            X[:, 1].reshape(1000, 1000),
            y.reshape(1000, 1000),
            cmap="rainbow",
        )

    def plot_result(self, solver, colorbar=[0, 0, 0]):
        from matplotlib import pyplot as plt

        X = np.array(
            [
                [x1, x2]
                for x1 in np.linspace(0, 1, 1000)
                for x2 in np.linspace(0, 1, 1000)
            ]
        )
        y = self.sol(X)
        model_y = solver.model.predict(X)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axs = []
        axs.append(
            self.plot_heatmap_at_axes(X, y, axes=axes[0], title="Exact solution")
        )
        axs.append(self.plot_heatmap_at_axes(X, model_y, axes[1], title=solver.name))
        axs.append(
            self.plot_heatmap_at_axes(
                X, np.abs(model_y - y), axes[2], title="Absolute error"
            )
        )

        for needColorbar, ax, axe in zip(colorbar, axs, axes):
            if needColorbar:
                fig.colorbar(ax, ax=axe)
        plt.show()
        return fig, axes


class Helmholtz_Hole(PDECases):
    def __init__(
        self,
        precision_train=15,
        precision_test=30,
        hard_constraint=False,
        layer_size=[2] + [350] * 3 + [1],
        activation="sin",
        initializer="Glorot uniform",
    ):
        self.n = 1
        self.k0 = 2 * np.pi * self.n
        self.hard_constraint = hard_constraint
        self.NumDomain, self.NumBoundary, self.NumTest = self.get_NumDomain(
            precision_train, precision_test
        )

        R = 1 / 4
        length = 1
        self.inner = dde.geometry.Disk([0, 0], R)
        self.outer = dde.geometry.Rectangle(
            [-length / 2, -length / 2], [length / 2, length / 2]
        )

        super().__init__(
            name="Helmholtz equation over a 2D square domain with a hole",
            NumDomain=self.NumDomain,
            use_output_transform=False,
            layer_size=layer_size,
            activation=activation,
            initializer=initializer,
        )

    def get_NumDomain(self, precision_train, precision_test):
        wave_len = 1 / self.n

        hx_train = wave_len / precision_train
        nx_train = int(1 / hx_train)

        hx_test = wave_len / precision_test
        nx_test = int(1 / hx_test)

        num_domain = nx_train**2
        num_boundary = 4 * nx_train
        num_test = nx_test**2
        return num_domain, num_boundary, num_test

    def gen_pde(self):
        def pde(x, y):
            dy_xx = dde.grad.hessian(y, x, i=0, j=0)
            dy_yy = dde.grad.hessian(y, x, i=1, j=1)

            f = (
                self.k0**2
                * bkd.sin(self.k0 * x[:, 0:1])
                * bkd.sin(self.k0 * x[:, 1:2])
            )
            return -dy_xx - dy_yy - self.k0**2 * y - f

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

        return dde.data.PDE(
            self.geomtime,
            self.pde,
            [bc_inner, bc_outer],
            num_domain=self.NumDomain,
            num_boundary=self.NumBoundary,
            solution=self.sol,
            num_test=self.NumTest,
        )

    def set_axes(self, axes):
        axes.set_xlim(-1 / 2, 1 / 2)
        axes.set_ylim(-1 / 2, 1 / 2)
        axes.set_xlabel("x1")
        axes.set_ylabel("x2")

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
        return axes.pcolormesh(
            X[:, 0].reshape(1000, 1000),
            X[:, 1].reshape(1000, 1000),
            y.reshape(1000, 1000),
            cmap="rainbow",
        )

    def plot_result(self, solver, colorbar=[0, 0, 0]):
        from matplotlib import pyplot as plt

        X = np.array(
            [
                [x1, x2]
                for x1 in np.linspace(-1 / 2, 1 / 2, 1000)
                for x2 in np.linspace(-1 / 2, 1 / 2, 1000)
            ]
        )
        y = self.sol(X)
        y[self.geomtime.inside(X) == 0] = np.nan
        model_y = solver.model.predict(X)
        model_y[self.geomtime.inside(X) == 0] = np.nan

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axs = []
        axs.append(
            self.plot_heatmap_at_axes(X, y, axes=axes[0], title="Exact solution")
        )
        axs.append(self.plot_heatmap_at_axes(X, model_y, axes[1], title=solver.name))
        axs.append(
            self.plot_heatmap_at_axes(
                X, np.abs(model_y - y), axes[2], title="Absolute error"
            )
        )

        for needColorbar, ax, axe in zip(colorbar, axs, axes):
            if needColorbar:
                fig.colorbar(ax, ax=axe)
        plt.show()
        return fig, axes


class Helmholtz_Sound_hard_Absorbing(PDECases):
    def __init__(
        self,
        hard_constraint=False,
        layer_size=[2] + [350] * 3 + [2],
        activation="tanh",
        initializer="Glorot uniform",
    ):
        self.R = np.pi / 4
        self.length = 2 * np.pi
        self.inner = dde.geometry.Disk([0, 0], self.R)
        self.outer = dde.geometry.Rectangle(
            [-self.length / 2, -self.length / 2], [self.length / 2, self.length / 2]
        )

        self.n = 1
        self.k0 = 2 * np.pi * self.n
        self.hard_constraint = hard_constraint
        self.NumDomain, self.NumBoundary, self.NumTest = self.get_NumDomain()

        super().__init__(
            name="Helmholtz sound-hard scattering problem with absorbing boundary conditions",
            NumDomain=self.NumDomain,
            use_output_transform=False,
            layer_size=layer_size,
            activation=activation,
            initializer=initializer,
        )

    def get_NumDomain(self):
        wave_len = 2 * np.pi / self.k0
        n_wave = 20
        h_elem = wave_len / n_wave
        nx = int(self.length / h_elem)

        num_domain = nx**2
        num_boundary = 8 * nx
        num_test = 5 * nx**2
        return num_domain, num_boundary, num_test

    def gen_pde(self):
        def pde(x, y):
            y0, y1 = y[:, 0:1], y[:, 1:2]

            y0_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
            y0_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)

            y1_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
            y1_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)

            return [
                -y0_xx - y0_yy - self.k0**2 * y0,
                -y1_xx - y1_yy - self.k0**2 * y1,
            ]

        return pde

    def sound_hard_circle_deepxde(self, k0, a, points):
        from scipy.special import jv, hankel1

        fem_xx = points[:, 0:1]
        fem_xy = points[:, 1:2]
        r = np.sqrt(fem_xx * fem_xx + fem_xy * fem_xy)
        theta = np.arctan2(fem_xy, fem_xx)
        npts = np.size(fem_xx, 0)
        n_terms = int(30 + (k0 * a) ** 1.01)
        u_sc = np.zeros((npts), dtype=np.complex128)
        for n in range(-n_terms, n_terms):
            bessel_deriv = jv(n - 1, k0 * a) - n / (k0 * a) * jv(n, k0 * a)
            hankel_deriv = n / (k0 * a) * hankel1(n, k0 * a) - hankel1(n + 1, k0 * a)
            u_sc += (
                -((1j) ** (n))
                * (bessel_deriv / hankel_deriv)
                * hankel1(n, k0 * r)
                * np.exp(1j * n * theta)
            ).ravel()
        return u_sc

    def sol(self, x):
        result = self.sound_hard_circle_deepxde(self.k0, self.R, x).reshape(
            (x.shape[0], 1)
        )
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

        bc0_inner = dde.NeumannBC(
            self.geomtime, func0_inner, boundary_inner, component=0
        )
        bc1_inner = dde.NeumannBC(
            self.geomtime, func1_inner, boundary_inner, component=1
        )

        bc0_outer = dde.RobinBC(self.geomtime, func0_outer, boundary_outer, component=0)
        bc1_outer = dde.RobinBC(self.geomtime, func1_outer, boundary_outer, component=1)

        bcs = [bc0_inner, bc1_inner, bc0_outer, bc1_outer]

        return dde.data.PDE(
            self.geomtime,
            self.pde,
            bcs,
            num_domain=self.NumDomain,
            num_boundary=self.NumBoundary,
            solution=self.sol,
            num_test=self.NumTest,
        )

    def set_axes(self, axes):
        axes.set_xlim(-self.length / 2, self.length / 2)
        axes.set_ylim(-self.length / 2, self.length / 2)
        axes.set_xlabel("x1")
        axes.set_ylabel("x2")

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
        return axes.pcolormesh(
            X[:, 0].reshape(1000, 1000),
            X[:, 1].reshape(1000, 1000),
            y.reshape(1000, 1000),
            cmap="rainbow",
        )

    def plot_result(self, solver, colorbar=[0, 0, 0]):
        from matplotlib import pyplot as plt

        X = np.array(
            [
                [x1, x2]
                for x1 in np.linspace(-self.length / 2, self.length / 2, 1000)
                for x2 in np.linspace(-self.length / 2, self.length / 2, 1000)
            ]
        )
        y = self.sol(X)[:, 0]
        y[self.geomtime.inside(X) == 0] = np.nan
        model_y = solver.model.predict(X)[:, 0]
        model_y[self.geomtime.inside(X) == 0] = np.nan

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axs = []
        axs.append(
            self.plot_heatmap_at_axes(X, y, axes=axes[0], title="Exact solution")
        )
        axs.append(self.plot_heatmap_at_axes(X, model_y, axes[1], title=solver.name))
        axs.append(
            self.plot_heatmap_at_axes(
                X, np.abs(model_y - y), axes[2], title="Absolute error"
            )
        )

        for needColorbar, ax, axe in zip(colorbar, axs, axes):
            if needColorbar:
                fig.colorbar(ax, ax=axe)
        plt.show()
        return fig, axes


class Kovasznay_Flow(PDECases):
    def __init__(
        self,
        NumDomain=2601,
        layer_size=[2] + [50] * 4 + [3],
        activation="tanh",
        initializer="Glorot normal",
    ):
        self.Re = 20
        self.nu = 1 / self.Re
        self.l = 1 / (2 * self.nu) - np.sqrt(1 / (4 * self.nu**2) + 4 * np.pi**2)
        super().__init__(
            name="Kovasznay flow",
            NumDomain=NumDomain,
            use_output_transform=False,
            layer_size=layer_size,
            activation=activation,
            initializer=initializer,
        )

    def gen_pde(self):
        def pde(x, u):
            u_vel, v_vel, p = u[:, 0:1], u[:, 1:2], u[:, 2:]
            u_vel_x = dde.grad.jacobian(u, x, i=0, j=0)
            u_vel_y = dde.grad.jacobian(u, x, i=0, j=1)
            u_vel_xx = dde.grad.hessian(u, x, component=0, i=0, j=0)
            u_vel_yy = dde.grad.hessian(u, x, component=0, i=1, j=1)

            v_vel_x = dde.grad.jacobian(u, x, i=1, j=0)
            v_vel_y = dde.grad.jacobian(u, x, i=1, j=1)
            v_vel_xx = dde.grad.hessian(u, x, component=1, i=0, j=0)
            v_vel_yy = dde.grad.hessian(u, x, component=1, i=1, j=1)

            p_x = dde.grad.jacobian(u, x, i=2, j=0)
            p_y = dde.grad.jacobian(u, x, i=2, j=1)

            momentum_x = (
                u_vel * u_vel_x
                + v_vel * u_vel_y
                + p_x
                - 1 / self.Re * (u_vel_xx + u_vel_yy)
            )
            momentum_y = (
                u_vel * v_vel_x
                + v_vel * v_vel_y
                + p_y
                - 1 / self.Re * (v_vel_xx + v_vel_yy)
            )
            continuity = u_vel_x + v_vel_y

            return [momentum_x, momentum_y, continuity]

        return pde

    def sol(self, x):
        u = 1 - np.exp(self.l * x[:, 0:1]) * np.cos(2 * np.pi * x[:, 1:2])
        v = (
            self.l
            / (2 * np.pi)
            * np.exp(self.l * x[:, 0:1])
            * np.sin(2 * np.pi * x[:, 1:2])
        )
        p = 1 / 2 * (1 - np.exp(2 * self.l * x[:, 0:1]))
        return np.hstack((u, v, p))

    def gen_geomtime(self):
        return dde.geometry.Rectangle(xmin=[-0.5, -0.5], xmax=[1, 1.5])

    def gen_data(self):
        def u_func(x):
            return 1 - np.exp(self.l * x[:, 0:1]) * np.cos(2 * np.pi * x[:, 1:2])

        def v_func(x):
            return (
                self.l
                / (2 * np.pi)
                * np.exp(self.l * x[:, 0:1])
                * np.sin(2 * np.pi * x[:, 1:2])
            )

        def p_func(x):
            return 1 / 2 * (1 - np.exp(2 * self.l * x[:, 0:1]))

        def boundary_outflow(x, on_boundary):
            return on_boundary and np.isclose(x[0], 1)

        boundary_condition_u = dde.icbc.DirichletBC(
            self.geomtime,
            lambda x: self.sol(x)[:, 0],
            lambda _, on_boundary: on_boundary,
            component=0,
        )
        boundary_condition_v = dde.icbc.DirichletBC(
            self.geomtime,
            lambda x: self.sol(x)[:, 1],
            lambda _, on_boundary: on_boundary,
            component=1,
        )
        boundary_condition_right_p = dde.icbc.DirichletBC(
            self.geomtime, lambda x: self.sol(x)[:, 2], boundary_outflow, component=2
        )
        return dde.data.PDE(
            self.geomtime,
            self.pde,
            [boundary_condition_u, boundary_condition_v, boundary_condition_right_p],
            num_domain=self.NumDomain,
            num_boundary=400,
            solution=self.sol,
            num_test=100000,
        )

    def set_axes(self, axes):
        axes.set_xlim(-0.5, 1)
        axes.set_ylim(-0.5, 1.5)
        axes.set_xlabel("x1")
        axes.set_ylabel("x2")

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
        return axes.pcolormesh(
            X[:, 0].reshape(1000, 1000),
            X[:, 1].reshape(1000, 1000),
            y.reshape(1000, 1000),
            cmap="rainbow",
        )

    def plot_result(self, solver, colorbar=[0, 0, 0]):
        from matplotlib import pyplot as plt

        X = np.array(
            [
                [x1, x2]
                for x1 in np.linspace(-0.5, 1, 1000)
                for x2 in np.linspace(-0.5, 1.5, 1000)
            ]
        )
        y = self.sol(X)[:, 0]
        y[self.geomtime.inside(X) == 0] = np.nan
        model_y = solver.model.predict(X)[:, 0]
        model_y[self.geomtime.inside(X) == 0] = np.nan

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axs = []
        axs.append(
            self.plot_heatmap_at_axes(X, y, axes=axes[0], title="Exact solution")
        )
        axs.append(self.plot_heatmap_at_axes(X, model_y, axes[1], title=solver.name))
        axs.append(
            self.plot_heatmap_at_axes(
                X, np.abs(model_y - y), axes[2], title="Absolute error"
            )
        )

        for needColorbar, ax, axe in zip(colorbar, axs, axes):
            if needColorbar:
                fig.colorbar(ax, ax=axe)
        plt.show()
        return fig, axes


class Euler_Beam(PDECases):
    def __init__(
        self,
        NumDomain=10,
        layer_size=[1] + [20] * 3 + [1],
        activation="tanh",
        initializer="Glorot normal",
    ):
        self.Re = 20
        self.nu = 1 / self.Re
        self.l = 1 / (2 * self.nu) - np.sqrt(1 / (4 * self.nu**2) + 4 * np.pi**2)
        super().__init__(
            name="Euler beam",
            NumDomain=NumDomain,
            use_output_transform=False,
            layer_size=layer_size,
            activation=activation,
            initializer=initializer,
        )

    def ddy(self, x, y):
        return dde.grad.hessian(y, x)

    def dddy(self, x, y):
        return dde.grad.jacobian(self.ddy(x, y), x)

    def gen_pde(self):
        def pde(x, y):
            dy_xx = self.ddy(x, y)
            dy_xxxx = dde.grad.hessian(dy_xx, x)
            return dy_xxxx + 1

        return pde

    def sol(self, x):
        return -(x**4) / 24 + x**3 / 6 - x**2 / 4

    def gen_geomtime(self):
        return dde.geometry.Interval(0, 1)

    def gen_data(self):
        def boundary_l(x, on_boundary):
            return on_boundary and np.isclose(x[0], 0)

        def boundary_r(x, on_boundary):
            return on_boundary and np.isclose(x[0], 1)

        bc1 = dde.icbc.DirichletBC(self.geomtime, lambda x: 0, boundary_l)
        bc2 = dde.icbc.NeumannBC(self.geomtime, lambda x: 0, boundary_l)
        bc3 = dde.icbc.OperatorBC(
            self.geomtime, lambda x, y, _: self.ddy(x, y), boundary_r
        )
        bc4 = dde.icbc.OperatorBC(
            self.geomtime, lambda x, y, _: self.dddy(x, y), boundary_r
        )
        return dde.data.PDE(
            self.geomtime,
            self.pde,
            [bc1, bc2, bc3, bc4],
            num_domain=self.NumDomain,
            num_boundary=2,
            solution=self.sol,
            num_test=100,
        )

    def plot_result(self, solver, axes=None, exact=True):
        from matplotlib import pyplot as plt

        X, y = self.gen_testdata()
        if axes is None:
            fig, axes = plt.subplots()
        if exact:
            axes.plot(X, y, label="Exact")
        axes.plot(X, solver.model.predict(X), "--", label="Prediction")
        axes.legend()
        axes.set_xlabel("t")
        axes.set_ylabel("y")
        axes.set_title(self.name)
        return fig, axes


class Heat(PDECases):
    def __init__(
        self,
        NumDomain=2540,
        layer_size=[2] + [20] * 3 + [1],
        activation="tanh",
        initializer="Glorot normal",
    ):
        self.a = 0.4
        self.L = 1
        self.n = 1
        super().__init__(
            name="Heat equation",
            NumDomain=NumDomain,
            use_output_transform=False,
            layer_size=layer_size,
            activation=activation,
            initializer=initializer,
        )

    def gen_pde(self):
        def pde(x, y):
            dy_t = dde.grad.jacobian(y, x, i=0, j=1)
            dy_xx = dde.grad.hessian(y, x, i=0, j=0)
            return dy_t - self.a * dy_xx

        return pde

    def sol(self, X):
        x = X[:, 0:1]
        t = X[:, 1:2]
        return np.exp(
            -(self.n**2 * np.pi**2 * self.a * t) / (self.L**2)
        ) * np.sin(self.n * np.pi * x / self.L)

    def gen_geomtime(self):
        geom = dde.geometry.Interval(0, self.L)
        timedomain = dde.geometry.TimeDomain(0, 1)
        return dde.geometry.GeometryXTime(geom, timedomain)

    def gen_data(self):
        bc = dde.icbc.DirichletBC(
            self.geomtime, lambda x: 0, lambda _, on_boundary: on_boundary
        )
        ic = dde.icbc.IC(
            self.geomtime,
            lambda x: np.sin(self.n * np.pi * x[:, 0:1] / self.L),
            lambda _, on_initial: on_initial,
        )
        return dde.data.TimePDE(
            self.geomtime,
            self.pde,
            [bc, ic],
            num_domain=self.NumDomain,
            num_boundary=80,
            solution=self.sol,
            num_test=2540,
            num_initial=160,
        )

    def set_axes(self, axes):
        axes.set_xlim(0, 1)
        axes.set_ylim(0, self.L)
        axes.set_xlabel("t")
        axes.set_ylabel("x")

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
        return axes.pcolormesh(
            X[:, 1].reshape(1000, 1000),
            X[:, 0].reshape(1000, 1000),
            y.reshape(1000, 1000),
            cmap="rainbow",
        )

    def plot_result(self, solver, colorbar=[0, 0, 0]):
        from matplotlib import pyplot as plt

        X = np.array(
            [
                [x, t]
                for x in np.linspace(0, self.L, 1000)
                for t in np.linspace(0, 1, 1000)
            ]
        )
        y = self.sol(X)
        model_y = solver.model.predict(X)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axs = []
        axs.append(
            self.plot_heatmap_at_axes(X, y, axes=axes[0], title="Exact solution")
        )
        axs.append(self.plot_heatmap_at_axes(X, model_y, axes[1], title=solver.name))
        axs.append(
            self.plot_heatmap_at_axes(
                X, np.abs(model_y - y), axes[2], title="Absolute error"
            )
        )

        for needColorbar, ax, axe in zip(colorbar, axs, axes):
            if needColorbar:
                fig.colorbar(ax, ax=axe)
        plt.show()
        return fig, axes


class Klein_Gordon(PDECases):
    def __init__(
        self,
        NumDomain=30000,
        layer_size=[2] + [40] * 2 + [1],
        activation="tanh",
        initializer="Glorot normal",
    ):
        self.a = 0.4
        self.L = 1
        self.n = 1
        super().__init__(
            name="Klein-Gordon equation",
            NumDomain=NumDomain,
            use_output_transform=False,
            layer_size=layer_size,
            activation=activation,
            initializer=initializer,
        )

    def gen_pde(self):
        def pde(x, y):
            alpha, beta, gamma, k = -1, 0, 1, 2
            dy_tt = dde.grad.hessian(y, x, i=1, j=1)
            dy_xx = dde.grad.hessian(y, x, i=0, j=0)
            x, t = x[:, 0:1], x[:, 1:2]
            return (
                dy_tt
                + alpha * dy_xx
                + beta * y
                + gamma * (y**k)
                + x * bkd.cos(t)
                - (x**2) * (bkd.cos(t) ** 2)
            )

        return pde

    def sol(self, x):
        return x[:, 0:1] * np.cos(x[:, 1:2])

    def gen_geomtime(self):
        geom = dde.geometry.Interval(-1, 1)
        timedomain = dde.geometry.TimeDomain(0, 10)
        return dde.geometry.GeometryXTime(geom, timedomain)

    def gen_data(self):
        bc = dde.icbc.DirichletBC(
            self.geomtime, self.sol, lambda _, on_boundary: on_boundary
        )
        ic_1 = dde.icbc.IC(self.geomtime, self.sol, lambda _, on_initial: on_initial)
        ic_2 = dde.icbc.OperatorBC(
            self.geomtime,
            lambda x, y, _: dde.grad.jacobian(y, x, i=0, j=1),
            lambda _, on_initial: on_initial,
        )
        return dde.data.TimePDE(
            self.geomtime,
            self.pde,
            [bc, ic_1, ic_2],
            num_domain=self.NumDomain,
            num_boundary=1500,
            num_test=6000,
            num_initial=1500,
            solution=self.sol,
        )

    def set_axes(self, axes):
        axes.set_xlim(0, 10)
        axes.set_ylim(-1, 1)
        axes.set_xlabel("t")
        axes.set_ylabel("x")

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
        return axes.pcolormesh(
            X[:, 1].reshape(1000, 1000),
            X[:, 0].reshape(1000, 1000),
            y.reshape(1000, 1000),
            cmap="rainbow",
        )

    def plot_result(self, solver, colorbar=[0, 0, 0]):
        from matplotlib import pyplot as plt

        X = np.array(
            [[x, t] for x in np.linspace(-1, 1, 1000) for t in np.linspace(0, 10, 1000)]
        )
        y = self.sol(X)
        model_y = solver.model.predict(X)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axs = []
        axs.append(
            self.plot_heatmap_at_axes(X, y, axes=axes[0], title="Exact solution")
        )
        axs.append(self.plot_heatmap_at_axes(X, model_y, axes[1], title=solver.name))
        axs.append(
            self.plot_heatmap_at_axes(
                X, np.abs(model_y - y), axes[2], title="Absolute error"
            )
        )

        for needColorbar, ax, axe in zip(colorbar, axs, axes):
            if needColorbar:
                fig.colorbar(ax, ax=axe)
        plt.show()
        return fig, axes


class Diffusion_reaction(PDECases):
    def __init__(
        self,
        NumDomain=320,
        layer_size=[2] + [30] * 6 + [1],
        activation="tanh",
        initializer="Glorot uniform",
    ):
        super().__init__(
            name="Diffusion-reaction equation",
            NumDomain=NumDomain,
            use_output_transform=True,
            layer_size=layer_size,
            activation=activation,
            initializer=initializer,
        )

    def gen_pde(self):
        def pde(x, y):
            dy_t = dde.grad.jacobian(y, x, i=0, j=1)
            dy_xx = dde.grad.hessian(y, x, i=0, j=0)
            d = 1
            return (
                dy_t
                - d * dy_xx
                - bkd.exp(-x[:, 1:])
                * (
                    3 * bkd.sin(2 * x[:, 0:1]) / 2
                    + 8 * bkd.sin(3 * x[:, 0:1]) / 3
                    + 15 * bkd.sin(4 * x[:, 0:1]) / 4
                    + 63 * bkd.sin(8 * x[:, 0:1]) / 8
                )
            )

        return pde

    def sol(self, x):
        return np.exp(-x[:, 1:]) * (
            np.sin(x[:, 0:1])
            + np.sin(2 * x[:, 0:1]) / 2
            + np.sin(3 * x[:, 0:1]) / 3
            + np.sin(4 * x[:, 0:1]) / 4
            + np.sin(8 * x[:, 0:1]) / 8
        )

    def gen_geomtime(self):
        geom = dde.geometry.Interval(-1, 1)
        timedomain = dde.geometry.TimeDomain(0, 0.99)
        return dde.geometry.GeometryXTime(geom, timedomain)

    def output_transform(self, x, y):
        return (
            x[:, 1:2] * (np.pi**2 - x[:, 0:1] ** 2) * y
            + bkd.sin(x[:, 0:1])
            + bkd.sin(2 * x[:, 0:1]) / 2
            + bkd.sin(3 * x[:, 0:1]) / 3
            + bkd.sin(4 * x[:, 0:1]) / 4
            + bkd.sin(8 * x[:, 0:1]) / 8
        )

    def gen_data(self):
        return dde.data.TimePDE(
            self.geomtime,
            self.pde,
            [],
            num_domain=self.NumDomain,
            num_test=80000,
            solution=self.sol,
        )

    def set_axes(self, axes):
        axes.set_xlim(0, 0.99)
        axes.set_ylim(-1, 1)
        axes.set_xlabel("t")
        axes.set_ylabel("x")

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
        return axes.pcolormesh(
            X[:, 1].reshape(1000, 1000),
            X[:, 0].reshape(1000, 1000),
            y.reshape(1000, 1000),
            cmap="rainbow",
        )

    def plot_result(self, solver, colorbar=[0, 0, 0]):
        from matplotlib import pyplot as plt

        X = np.array(
            [
                [x, t]
                for x in np.linspace(-1, 1, 1000)
                for t in np.linspace(0, 0.99, 1000)
            ]
        )
        y = self.sol(X)
        model_y = solver.model.predict(X)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axs = []
        axs.append(
            self.plot_heatmap_at_axes(X, y, axes=axes[0], title="Exact solution")
        )
        axs.append(self.plot_heatmap_at_axes(X, model_y, axes[1], title=solver.name))
        axs.append(
            self.plot_heatmap_at_axes(
                X, np.abs(model_y - y), axes[2], title="Absolute error"
            )
        )

        for needColorbar, ax, axe in zip(colorbar, axs, axes):
            if needColorbar:
                fig.colorbar(ax, ax=axe)
        plt.show()
        return fig, axes


class IDE(PDECases):
    def __init__(
        self,
        NumDomain=16,
        layer_size=[1] + [20] * 3 + [1],
        activation="tanh",
        initializer="Glorot uniform",
    ):
        super().__init__(
            name="Integro-differential equation",
            NumDomain=NumDomain,
            use_output_transform=False,
            layer_size=layer_size,
            activation=activation,
            initializer=initializer,
        )

    def gen_pde(self):
        def ide(x, y, int_mat):
            """int_0^x y(t)dt"""
            lhs1 = bkd.matmul(bkd.from_numpy(int_mat), bkd.from_numpy(y))
            lhs2 = dde.grad.jacobian(y, x, i=0, j=0)
            rhs = 2 * np.pi * bkd.cos(2 * np.pi * x) + bkd.sin(np.pi * x) ** 2 / np.pi
            return lhs1 + (lhs2 - rhs)[: bkd.size(lhs1)]

        return ide

    def sol(self, x):
        return np.sin(2 * np.pi * x)

    def gen_geomtime(self):
        return dde.geometry.TimeDomain(0, 1)

    def gen_data(self):
        quad_deg = 16
        ic = dde.icbc.IC(self.geomtime, self.sol, lambda _, on_initial: on_initial)
        return dde.data.IDE(
            self.geomtime,
            self.pde,
            ic,
            quad_deg,
            num_domain=self.NumDomain,
            num_boundary=2,
        )

    def plot_result(self, solver, axes=None, exact=True):
        from matplotlib import pyplot as plt

        X, y = self.gen_testdata()
        if axes is None:
            fig, axes = plt.subplots()
        if exact:
            axes.plot(X, y, label="Exact")
        axes.plot(X, solver.model.predict(X), "--", label="Prediction")
        axes.legend()
        axes.set_xlabel("t")
        axes.set_ylabel("y")
        axes.set_title(self.name)
        return fig, axes


class Volterra_IDE(PDECases):
    def __init__(
        self,
        NumDomain=10,
        layer_size=[1] + [20] * 3 + [1],
        activation="tanh",
        initializer="Glorot uniform",
    ):
        super().__init__(
            name="Volterra IDE",
            NumDomain=NumDomain,
            use_output_transform=False,
            layer_size=layer_size,
            activation=activation,
            initializer=initializer,
        )

    def gen_pde(self):
        def ide(x, y, int_mat):
            """int_0^x y(t)dt"""
            rhs = bkd.matmul(bkd.from_numpy(int_mat), bkd.from_numpy(y))
            lhs1 = dde.grad.jacobian(y, x, i=0, j=0)

            return (lhs1 + y)[: bkd.size(rhs)] - rhs

        return ide

    def sol(self, x):
        return np.exp(-x) * np.cosh(x)

    def gen_geomtime(self):
        return dde.geometry.TimeDomain(0, 5)

    def gen_data(self):
        quad_deg = 20
        ic = dde.icbc.IC(self.geomtime, self.sol, lambda _, on_initial: on_initial)

        def kernel(x, s):
            return np.exp(s - x)

        return dde.data.IDE(
            self.geomtime,
            self.pde,
            ic,
            quad_deg,
            kernel=kernel,
            num_domain=self.NumDomain,
            num_boundary=2,
            train_distribution="uniform",
        )

    def plot_result(self, solver, axes=None, exact=True):
        from matplotlib import pyplot as plt

        X, y = self.gen_testdata()
        if axes is None:
            fig, axes = plt.subplots()
        if exact:
            axes.plot(X, y, label="Exact")
        axes.plot(X, solver.model.predict(X), "--", label="Prediction")
        axes.legend()
        axes.set_xlabel("t")
        axes.set_ylabel("y")
        axes.set_title(self.name)
        return fig, axes


class Fractional_Poisson_1D(PDECases):
    def __init__(
        self,
        NumDomain=101,
        layer_size=[1] + [20] * 4 + [1],
        activation="tanh",
        initializer="Glorot normal",
    ):
        self.alpha = 1.5
        super().__init__(
            name="Fractional Poisson equation in 1D",
            NumDomain=NumDomain,
            use_output_transform=True,
            layer_size=layer_size,
            activation=activation,
            initializer=initializer,
        )

    def gen_pde(self):
        from scipy.special import gamma

        def fpde(x, y, int_mat):
            """(D_{0+}^alpha + D_{1-}^alpha) u(x) = f(x)"""
            if isinstance(int_mat, (list, tuple)) and len(int_mat) == 3:
                int_mat = bkd.sparse_tensor(*int_mat)
                lhs = bkd.sparse_dense_matmul(int_mat, y)
            else:
                lhs = bkd.matmul(bkd.from_numpy(int_mat), bkd.from_numpy(y))
            rhs = (
                gamma(4)
                / gamma(4 - self.alpha)
                * (x ** (3 - self.alpha) + (1 - x) ** (3 - self.alpha))
                - 3
                * gamma(5)
                / gamma(5 - self.alpha)
                * (x ** (4 - self.alpha) + (1 - x) ** (4 - self.alpha))
                + 3
                * gamma(6)
                / gamma(6 - self.alpha)
                * (x ** (5 - self.alpha) + (1 - x) ** (5 - self.alpha))
                - gamma(7)
                / gamma(7 - self.alpha)
                * (x ** (6 - self.alpha) + (1 - x) ** (6 - self.alpha))
            )
            # lhs /= 2 * np.cos(alpha * np.pi / 2)
            # rhs = gamma(alpha + 2) * x
            return lhs - rhs[: bkd.size(lhs)]

        return fpde

    def sol(self, x):
        return x**3 * (1 - x) ** 3

    def gen_geomtime(self):
        return dde.geometry.Interval(0, 1)

    def gen_data(self):
        bc = dde.icbc.DirichletBC(
            self.geomtime, self.sol, lambda _, on_boundary: on_boundary
        )
        return dde.data.FPDE(
            self.geomtime,
            self.pde,
            self.alpha,
            bc,
            [self.NumDomain],
            meshtype="static",
            solution=self.sol,
        )

    def output_transform(self, x, y):
        return x * (1 - x) * y

    def plot_result(self, solver, axes=None, exact=True):
        from matplotlib import pyplot as plt

        X, y = self.gen_testdata()
        if axes is None:
            fig, axes = plt.subplots()
        if exact:
            axes.plot(X, y, label="Exact")
        axes.plot(X, solver.model.predict(X), "--", label="Prediction")
        axes.legend()
        axes.set_xlabel("x")
        axes.set_ylabel("y")
        axes.set_title(self.name)
        return fig, axes


class Fractional_Poisson_2D(PDECases):
    def __init__(
        self,
        NumDomain=100,
        layer_size=[2] + [20] * 4 + [1],
        activation="tanh",
        initializer="Glorot normal",
    ):
        self.alpha = 1.8
        super().__init__(
            name="Fractional Poisson equation in 2D",
            NumDomain=NumDomain,
            use_output_transform=True,
            layer_size=layer_size,
            activation=activation,
            initializer=initializer,
        )

    def gen_pde(self):
        from scipy.special import gamma

        def fpde(x, y, int_mat):
            """\int_theta D_theta^alpha u(x)"""
            if isinstance(int_mat, (list, tuple)) and len(int_mat) == 3:
                int_mat = bkd.sparse_tensor(*int_mat)
                lhs = bkd.sparse_dense_matmul(int_mat, y)
            else:
                lhs = bkd.matmul(bkd.from_numpy(int_mat), bkd.from_numpy(y))
            lhs = lhs[:, 0]
            lhs *= (
                gamma((1 - self.alpha) / 2)
                * gamma((2 + self.alpha) / 2)
                / (2 * np.pi**1.5)
            )
            x = x[: bkd.size(lhs)]
            rhs = (
                2**self.alpha
                * gamma(2 + self.alpha / 2)
                * gamma(1 + self.alpha / 2)
                * (
                    1
                    - (1 + self.alpha / 2)
                    * bkd.from_numpy(np.sum(bkd.to_numpy(x**2), axis=1))
                )
            )
            return lhs - rhs

        return fpde

    def sol(self, x):
        return (np.abs(1 - np.linalg.norm(x, axis=1, keepdims=True) ** 2)) ** (
            1 + self.alpha / 2
        )

    def gen_geomtime(self):
        return dde.geometry.Disk([0, 0], 1)

    def gen_data(self):
        bc = dde.icbc.DirichletBC(
            self.geomtime, self.sol, lambda _, on_boundary: on_boundary
        )
        return dde.data.FPDE(
            self.geomtime,
            self.pde,
            self.alpha,
            bc,
            [8, 100],
            num_domain=self.NumDomain,
            num_boundary=1,
            solution=self.sol,
        )

    def output_transform(self, x, y):
        return (
            1 - bkd.from_numpy(np.sum(bkd.to_numpy(x) ** 2, axis=1, keepdims=True))
        ) * y

    def set_axes(self, axes):
        axes.set_xlim(-1, 1)
        axes.set_ylim(-1, 1)
        axes.set_xlabel("x1")
        axes.set_ylabel("x2")

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
        return axes.pcolormesh(
            X[:, 0].reshape(1000, 1000),
            X[:, 1].reshape(1000, 1000),
            y.reshape(1000, 1000),
            cmap="rainbow",
        )

    def plot_result(self, solver, colorbar=[0, 0, 0]):
        from matplotlib import pyplot as plt

        X = np.array(
            [
                [x1, x2]
                for x1 in np.linspace(-1, 1, 1000)
                for x2 in np.linspace(-1, 1, 1000)
            ]
        )
        y = self.sol(X)
        y[self.geomtime.inside(X) == 0] = np.nan
        model_y = solver.model.predict(X)
        model_y[self.geomtime.inside(X) == 0] = np.nan

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axs = []
        axs.append(
            self.plot_heatmap_at_axes(X, y, axes=axes[0], title="Exact solution")
        )
        axs.append(self.plot_heatmap_at_axes(X, model_y, axes[1], title=solver.name))
        axs.append(
            self.plot_heatmap_at_axes(
                X, np.abs(model_y - y), axes[2], title="Absolute error"
            )
        )

        for needColorbar, ax, axe in zip(colorbar, axs, axes):
            if needColorbar:
                fig.colorbar(ax, ax=axe)
        plt.show()
        return fig, axes


class Fractional_Poisson_3D(PDECases):
    def __init__(
        self,
        NumDomain=256,
        layer_size=[3] + [20] * 4 + [1],
        activation="tanh",
        initializer="Glorot normal",
    ):
        self.alpha = 1.8
        super().__init__(
            name="Fractional Poisson equation in 3D",
            NumDomain=NumDomain,
            use_output_transform=True,
            layer_size=layer_size,
            activation=activation,
            initializer=initializer,
        )

    def gen_pde(self):
        from scipy.special import gamma

        def fpde(x, y, int_mat):
            """\int_theta D_theta^alpha u(x)"""
            if isinstance(int_mat, (list, tuple)) and len(int_mat) == 3:
                int_mat = bkd.sparse_tensor(*int_mat)
                lhs = bkd.sparse_dense_matmul(int_mat, y)
            else:
                lhs = bkd.matmul(int_mat, y)
            lhs = lhs[:, 0]
            lhs *= (
                gamma((1 - self.alpha) / 2)
                * gamma((3 + self.alpha) / 2)
                / (2 * np.pi**2)
            )
            x = x[: bkd.size(lhs)]
            rhs = (
                2**self.alpha
                * gamma(2 + self.alpha / 2)
                * gamma((3 + self.alpha) / 2)
                / gamma(3 / 2)
                * (
                    1
                    - (1 + self.alpha / 3)
                    * bkd.from_numpy(np.sum(bkd.to_numpy(x**2), axis=1))
                )
            )
            return lhs - rhs

        return fpde

    def sol(self, x):
        return (np.abs(1 - np.linalg.norm(x, axis=1, keepdims=True) ** 2)) ** (
            1 + self.alpha / 2
        )

    def gen_geomtime(self):
        return dde.geometry.Sphere([0, 0, 0], 1)

    def gen_data(self):
        bc = dde.icbc.DirichletBC(
            self.geomtime, self.sol, lambda _, on_boundary: on_boundary
        )
        return dde.data.FPDE(
            self.geomtime,
            self.pde,
            self.alpha,
            bc,
            [8, 8, 100],
            num_domain=self.NumDomain,
            num_boundary=1,
            solution=self.sol,
        )

    def output_transform(self, x, y):
        return (
            1 - bkd.from_numpy(np.sum(bkd.to_numpy(x) ** 2, axis=1, keepdims=True))
        ) * y

    def set_axes(self, axes):
        axes.set_xlim(-1, 1)
        axes.set_ylim(-1, 1)
        axes.set_xlabel("x1")
        axes.set_ylabel("x2")

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
        return axes.pcolormesh(
            X[:, 0].reshape(1000, 1000),
            X[:, 1].reshape(1000, 1000),
            y.reshape(1000, 1000),
            cmap="rainbow",
        )

    def plot_result(self, solver, colorbar=[0, 0, 0]):
        from matplotlib import pyplot as plt

        X = np.array(
            [
                [x1, x2]
                for x1 in np.linspace(-1, 1, 1000)
                for x2 in np.linspace(-1, 1, 1000)
            ]
        )
        x3 = 0
        X = np.concatenate([X, np.ones((X.shape[0], 1)) * x3], axis=1)
        y = self.sol(X)
        y[self.geomtime.inside(X) == 0] = np.nan
        model_y = solver.model.predict(X)
        model_y[self.geomtime.inside(X) == 0] = np.nan

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axs = []
        axs.append(
            self.plot_heatmap_at_axes(X, y, axes=axes[0], title="Exact solution")
        )
        axs.append(self.plot_heatmap_at_axes(X, model_y, axes[1], title=solver.name))
        axs.append(
            self.plot_heatmap_at_axes(
                X, np.abs(model_y - y), axes[2], title="Absolute error"
            )
        )

        for needColorbar, ax, axe in zip(colorbar, axs, axes):
            if needColorbar:
                fig.colorbar(ax, ax=axe)
        plt.show()
        return fig, axes


class Fractional_Diffusion_1D(PDECases):
    def __init__(
        self,
        NumDomain=400,
        Dynamic_auxiliary_points=False,
        layer_size=[2] + [20] * 4 + [1],
        activation="tanh",
        initializer="Glorot normal",
    ):
        self.alpha = 1.8
        self.Dynamics_auxiliary_points = Dynamic_auxiliary_points
        if self.Dynamics_auxiliary_points:
            NumDomain = 20
        super().__init__(
            name="Fractional Poisson equation in 1D",
            NumDomain=NumDomain,
            use_output_transform=True,
            layer_size=layer_size,
            activation=activation,
            initializer=initializer,
        )

    def gen_pde(self):
        from scipy.special import gamma

        def fpde(x, y, int_mat):
            """du/dt + (D_{0+}^alpha + D_{1-}^alpha) u(x) = f(x)"""
            if isinstance(int_mat, (list, tuple)) and len(int_mat) == 3:
                int_mat = bkd.sparse_tensor(*int_mat)
                lhs = bkd.sparse_dense_matmul(int_mat, y)
            else:
                lhs = bkd.matmul(bkd.from_numpy(int_mat), bkd.from_numpy(y))
            dy_t = dde.grad.jacobian(y, x, i=0, j=1)
            x, t = x[:, :-1], x[:, -1:]
            rhs = -dy_t - bkd.exp(-t) * (
                x**3 * (1 - x) ** 3
                + gamma(4)
                / gamma(4 - self.alpha)
                * (x ** (3 - self.alpha) + (1 - x) ** (3 - self.alpha))
                - 3
                * gamma(5)
                / gamma(5 - self.alpha)
                * (x ** (4 - self.alpha) + (1 - x) ** (4 - self.alpha))
                + 3
                * gamma(6)
                / gamma(6 - self.alpha)
                * (x ** (5 - self.alpha) + (1 - x) ** (5 - self.alpha))
                - gamma(7)
                / gamma(7 - self.alpha)
                * (x ** (6 - self.alpha) + (1 - x) ** (6 - self.alpha))
            )
            return lhs - rhs[: bkd.size(lhs)]

        return fpde

    def sol(self, x):
        x, t = x[:, :-1], x[:, -1:]
        return np.exp(-t) * x**3 * (1 - x) ** 3

    def gen_geomtime(self):
        geom = dde.geometry.Interval(0, 1)
        timedomain = dde.geometry.TimeDomain(0, 1)
        return dde.geometry.GeometryXTime(geom, timedomain)

    def gen_data(self):
        bc = dde.icbc.DirichletBC(
            self.geomtime, self.sol, lambda _, on_boundary: on_boundary
        )
        ic = dde.icbc.IC(self.geomtime, self.sol, lambda _, on_initial: on_initial)
        # Dynamic auxiliary points
        if self.Dynamics_auxiliary_points:
            data = dde.data.TimeFPDE(
                self.geomtime,
                self.pde,
                self.alpha,
                [bc, ic],
                [100],
                num_domain=self.NumDomain,
                num_boundary=1,
                num_initial=1,
                num_test=50,
                solution=self.sol,
            )
        else:
            data = dde.data.TimeFPDE(
                self.geomtime,
                self.pde,
                self.alpha,
                [bc, ic],
                [52],
                num_domain=self.NumDomain,
                meshtype="static",
                solution=self.sol,
            )
        return data

    def output_transform(self, x, y):
        return (
            x[:, 0:1] * (1 - x[:, 0:1]) * x[:, 1:2] * y
            + x[:, 0:1] ** 3 * (1 - x[:, 0:1]) ** 3
        )

    def set_axes(self, axes):
        axes.set_xlim(0, 1)
        axes.set_ylim(0, 1)
        axes.set_xlabel("t")
        axes.set_ylabel("x")

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
        return axes.pcolormesh(
            X[:, 1].reshape(1000, 1000),
            X[:, 0].reshape(1000, 1000),
            y.reshape(1000, 1000),
            cmap="rainbow",
        )

    def plot_result(self, solver, colorbar=[0, 0, 0]):
        from matplotlib import pyplot as plt

        X = np.array(
            [
                [x1, x2]
                for x1 in np.linspace(0, 1, 1000)
                for x2 in np.linspace(0, 1, 1000)
            ]
        )
        y = self.sol(X)
        model_y = solver.model.predict(X)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axs = []
        axs.append(
            self.plot_heatmap_at_axes(X, y, axes=axes[0], title="Exact solution")
        )
        axs.append(self.plot_heatmap_at_axes(X, model_y, axes[1], title=solver.name))
        axs.append(
            self.plot_heatmap_at_axes(
                X, np.abs(model_y - y), axes[2], title="Absolute error"
            )
        )

        for needColorbar, ax, axe in zip(colorbar, axs, axes):
            if needColorbar:
                fig.colorbar(ax, ax=axe)
        plt.show()
        return fig, axes


class Beltrami_flow(PDECases):
    def __init__(
        self,
        NumDomain=50000,
        layer_size=[4] + [50] * 4 + [4],
        activation="tanh",
        initializer="Glorot normal",
    ):
        self.a = 1
        self.d = 1
        self.Re = 1
        super().__init__(
            name="Beltrami flow",
            NumDomain=NumDomain,
            use_output_transform=False,
            layer_size=layer_size,
            activation=activation,
            initializer=initializer,
        )

    def gen_pde(self):
        def pde(x, u):
            u_vel, v_vel, w_vel, p = u[:, 0:1], u[:, 1:2], u[:, 2:3], u[:, 3:4]
            u_vel_x = dde.grad.jacobian(u, x, i=0, j=0)
            u_vel_y = dde.grad.jacobian(u, x, i=0, j=1)
            u_vel_z = dde.grad.jacobian(u, x, i=0, j=2)
            u_vel_t = dde.grad.jacobian(u, x, i=0, j=3)
            u_vel_xx = dde.grad.hessian(u, x, component=0, i=0, j=0)
            u_vel_yy = dde.grad.hessian(u, x, component=0, i=1, j=1)
            u_vel_zz = dde.grad.hessian(u, x, component=0, i=2, j=2)

            v_vel_x = dde.grad.jacobian(u, x, i=1, j=0)
            v_vel_y = dde.grad.jacobian(u, x, i=1, j=1)
            v_vel_z = dde.grad.jacobian(u, x, i=1, j=2)
            v_vel_t = dde.grad.jacobian(u, x, i=1, j=3)
            v_vel_xx = dde.grad.hessian(u, x, component=1, i=0, j=0)
            v_vel_yy = dde.grad.hessian(u, x, component=1, i=1, j=1)
            v_vel_zz = dde.grad.hessian(u, x, component=1, i=2, j=2)

            w_vel_x = dde.grad.jacobian(u, x, i=2, j=0)
            w_vel_y = dde.grad.jacobian(u, x, i=2, j=1)
            w_vel_z = dde.grad.jacobian(u, x, i=2, j=2)
            w_vel_t = dde.grad.jacobian(u, x, i=2, j=3)
            w_vel_xx = dde.grad.hessian(u, x, component=2, i=0, j=0)
            w_vel_yy = dde.grad.hessian(u, x, component=2, i=1, j=1)
            w_vel_zz = dde.grad.hessian(u, x, component=2, i=2, j=2)

            p_x = dde.grad.jacobian(u, x, i=3, j=0)
            p_y = dde.grad.jacobian(u, x, i=3, j=1)
            p_z = dde.grad.jacobian(u, x, i=3, j=2)

            momentum_x = (
                u_vel_t
                + (u_vel * u_vel_x + v_vel * u_vel_y + w_vel * u_vel_z)
                + p_x
                - 1 / self.Re * (u_vel_xx + u_vel_yy + u_vel_zz)
            )
            momentum_y = (
                v_vel_t
                + (u_vel * v_vel_x + v_vel * v_vel_y + w_vel * v_vel_z)
                + p_y
                - 1 / self.Re * (v_vel_xx + v_vel_yy + v_vel_zz)
            )
            momentum_z = (
                w_vel_t
                + (u_vel * w_vel_x + v_vel * w_vel_y + w_vel * w_vel_z)
                + p_z
                - 1 / self.Re * (w_vel_xx + w_vel_yy + w_vel_zz)
            )
            continuity = u_vel_x + v_vel_y + w_vel_z
            return [momentum_x, momentum_y, momentum_z, continuity]

        return pde

    def gen_geomtime(self):
        spatial_domain = dde.geometry.Cuboid(xmin=[-1, -1, -1], xmax=[1, 1, 1])
        temporal_domain = dde.geometry.TimeDomain(0, 1)
        return dde.geometry.GeometryXTime(spatial_domain, temporal_domain)

    def u_func(self, x):
        return (
            -self.a
            * (
                np.exp(self.a * x[:, 0:1])
                * np.sin(self.a * x[:, 1:2] + self.d * x[:, 2:3])
                + np.exp(self.a * x[:, 2:3])
                * np.cos(self.a * x[:, 0:1] + self.d * x[:, 1:2])
            )
            * np.exp(-(self.d**2) * x[:, 3:4])
        )

    def v_func(self, x):
        return (
            -self.a
            * (
                np.exp(self.a * x[:, 1:2])
                * np.sin(self.a * x[:, 2:3] + self.d * x[:, 0:1])
                + np.exp(self.a * x[:, 0:1])
                * np.cos(self.a * x[:, 1:2] + self.d * x[:, 2:3])
            )
            * np.exp(-(self.d**2) * x[:, 3:4])
        )

    def w_func(self, x):
        return (
            -self.a
            * (
                np.exp(self.a * x[:, 2:3])
                * np.sin(self.a * x[:, 0:1] + self.d * x[:, 1:2])
                + np.exp(self.a * x[:, 1:2])
                * np.cos(self.a * x[:, 2:3] + self.d * x[:, 0:1])
            )
            * np.exp(-(self.d**2) * x[:, 3:4])
        )

    def p_func(self, x):
        return (
            -0.5
            * self.a**2
            * (
                np.exp(2 * self.a * x[:, 0:1])
                + np.exp(2 * self.a * x[:, 1:2])
                + np.exp(2 * self.a * x[:, 2:3])
                + 2
                * np.sin(self.a * x[:, 0:1] + self.d * x[:, 1:2])
                * np.cos(self.a * x[:, 2:3] + self.d * x[:, 0:1])
                * np.exp(self.a * (x[:, 1:2] + x[:, 2:3]))
                + 2
                * np.sin(self.a * x[:, 1:2] + self.d * x[:, 2:3])
                * np.cos(self.a * x[:, 0:1] + self.d * x[:, 1:2])
                * np.exp(self.a * (x[:, 2:3] + x[:, 0:1]))
                + 2
                * np.sin(self.a * x[:, 2:3] + self.d * x[:, 0:1])
                * np.cos(self.a * x[:, 1:2] + self.d * x[:, 2:3])
                * np.exp(self.a * (x[:, 0:1] + x[:, 1:2]))
            )
            * np.exp(-2 * self.d**2 * x[:, 3:4])
        )

    def sol(self, x):
        return np.concatenate(
            (
                self.u_func(x),
                self.v_func(x),
                self.w_func(x),
                self.p_func(x),
            ),
            axis=1,
        )

    def gen_data(self):
        bc_u = dde.icbc.DirichletBC(
            self.geomtime, self.u_func, lambda _, on_boundary: on_boundary, component=0
        )
        bc_v = dde.icbc.DirichletBC(
            self.geomtime, self.v_func, lambda _, on_boundary: on_boundary, component=1
        )
        bc_w = dde.icbc.DirichletBC(
            self.geomtime, self.w_func, lambda _, on_boundary: on_boundary, component=2
        )

        ic_u = dde.icbc.IC(
            self.geomtime, self.u_func, lambda _, on_initial: on_initial, component=0
        )
        ic_v = dde.icbc.IC(
            self.geomtime, self.v_func, lambda _, on_initial: on_initial, component=1
        )
        ic_w = dde.icbc.IC(
            self.geomtime, self.w_func, lambda _, on_initial: on_initial, component=2
        )
        return dde.data.TimePDE(
            self.geomtime,
            self.pde,
            [bc_u, bc_v, bc_w, ic_u, ic_v, ic_w],
            num_domain=self.NumDomain,
            num_boundary=5000,
            num_initial=5000,
            num_test=10000,
        )

    def set_axes(self, axes):
        axes.set_xlim(0, 1)
        axes.set_ylim(-1, 1)
        axes.set_xlabel("t")
        axes.set_ylabel("x")

    def plot_data(self, X, axes=None):
        from matplotlib import pyplot as plt

        if axes is None:
            fig, axes = plt.subplots()
        self.set_axes(axes)
        axes.scatter(X[:, -1], X[:, 2])
        return axes

    def plot_heatmap_at_axes(self, X, y, axes, title):
        axes.set_title(title)
        self.set_axes(axes)
        return axes.pcolormesh(
            X[:, -1].reshape(1000, 1000),
            X[:, 2].reshape(1000, 1000),
            y.reshape(1000, 1000),
            cmap="rainbow",
        )

    def plot_result(self, solver, colorbar=[0, 0, 0]):
        from matplotlib import pyplot as plt

        X = np.array(
            [
                [0, 0, x, t]
                for x in np.linspace(-1, 1, 1000)
                for t in np.linspace(0, 1, 1000)
            ]
        )
        y = self.sol(X)[:, 0]
        model_y = solver.model.predict(X)[:, 0]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axs = []
        axs.append(
            self.plot_heatmap_at_axes(X, y, axes=axes[0], title="Exact solution")
        )
        axs.append(self.plot_heatmap_at_axes(X, model_y, axes[1], title=solver.name))
        axs.append(
            self.plot_heatmap_at_axes(
                X, np.abs(model_y - y), axes[2], title="Absolute error"
            )
        )

        for needColorbar, ax, axe in zip(colorbar, axs, axes):
            if needColorbar:
                fig.colorbar(ax, ax=axe)
        plt.show()
        return fig, axes


class Schrodinger(PDECases):
    def __init__(
        self,
        NumDomain=10000,
        layer_size=[2] + [100] * 4 + [2],
        activation="tanh",
        initializer="Glorot normal",
    ):
        super().__init__(
            name="Schrodinger equation",
            NumDomain=NumDomain,
            use_output_transform=False,
            layer_size=layer_size,
            activation=activation,
            initializer=initializer,
        )

    def gen_pde(self):
        def pde(x, y):
            u = y[:, 0:1]
            v = y[:, 1:2]

            u_t = dde.grad.jacobian(y, x, i=0, j=1)
            v_t = dde.grad.jacobian(y, x, i=1, j=1)

            u_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
            v_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)

            f_u = u_t + 0.5 * v_xx + (u**2 + v**2) * v
            f_v = v_t - 0.5 * u_xx - (u**2 + v**2) * u

            return [f_u, f_v]

        return pde

    def gen_testdata(self):
        import os
        from scipy.io import loadmat

        basepath = os.path.abspath(__file__)
        folder = os.path.dirname(os.path.dirname(basepath))
        data_path = os.path.join(folder, "data/NLS.mat")
        data = loadmat(data_path)
        t = data["tt"]
        x = data["x"]
        u = data["uu"]
        dt = dx = 0.01
        xx, tt = np.meshgrid(x, t)
        X = np.vstack((np.ravel(xx), np.ravel(tt))).T
        y = u.T.flatten()[:, None]
        y = np.hstack((y.real, y.imag))
        return X, y

    def gen_geomtime(self):
        geom = dde.geometry.Interval(-5, 5)
        timedomain = dde.geometry.TimeDomain(0, np.pi / 2)
        return dde.geometry.GeometryXTime(geom, timedomain)

    def gen_data(self):
        bc_u_0 = dde.icbc.PeriodicBC(
            self.geomtime,
            0,
            lambda _, on_boundary: on_boundary,
            derivative_order=0,
            component=0,
        )
        bc_u_1 = dde.icbc.PeriodicBC(
            self.geomtime,
            0,
            lambda _, on_boundary: on_boundary,
            derivative_order=1,
            component=0,
        )
        bc_v_0 = dde.icbc.PeriodicBC(
            self.geomtime,
            0,
            lambda _, on_boundary: on_boundary,
            derivative_order=0,
            component=1,
        )
        bc_v_1 = dde.icbc.PeriodicBC(
            self.geomtime,
            0,
            lambda _, on_boundary: on_boundary,
            derivative_order=1,
            component=1,
        )

        def init_cond_u(x):
            return 2 / np.cosh(x[:, 0:1])

        def init_cond_v(x):
            return 0

        ic_u = dde.icbc.IC(
            self.geomtime, init_cond_u, lambda _, on_initial: on_initial, component=0
        )
        ic_v = dde.icbc.IC(
            self.geomtime, init_cond_v, lambda _, on_initial: on_initial, component=1
        )
        return dde.data.TimePDE(
            self.geomtime,
            self.pde,
            [bc_u_0, bc_u_1, bc_v_0, bc_v_1, ic_u, ic_v],
            self.NumDomain,
            num_boundary=20,
            num_initial=200,
            train_distribution="pseudo",
        )

    def set_axes(self, axes):
        axes.set_xlim(0, np.pi / 2)
        axes.set_ylim(-5, 5)
        axes.set_xlabel("t")
        axes.set_ylabel("x")

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
        return axes.pcolormesh(
            X[:, 1].reshape(201, 256),
            X[:, 0].reshape(201, 256),
            y.reshape(201, 256),
            cmap="rainbow",
        )

    def plot_result(self, solver, colorbar=[0, 0, 0]):
        from matplotlib import pyplot as plt

        X, y = self.gen_testdata()
        model_y = solver.model.predict(X)

        y = y[:, 0]
        model_y = model_y[:, 0]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axs = []
        axs.append(
            self.plot_heatmap_at_axes(X, y, axes=axes[0], title="Exact solution")
        )
        axs.append(self.plot_heatmap_at_axes(X, model_y, axes[1], title=solver.name))
        axs.append(
            self.plot_heatmap_at_axes(
                X, np.abs(model_y - y), axes[2], title="Absolute error"
            )
        )

        for needColorbar, ax, axe in zip(colorbar, axs, axes):
            if needColorbar:
                fig.colorbar(ax, ax=axe)
        plt.show()
        return fig, axes
