import numpy as np
import deepxde as dde
import deepxde.backend as bkd
from abc import abstractmethod
from .PDECases import PDECases


class InverseCase(PDECases):
    def __init__(
        self,
        name,
        external_trainable_variables,
        NumDomain=2000,
        use_output_transform=False,
        layer_size=[2] + [32] * 3 + [1],
        activation="tanh",
        initializer="Glorot uniform",
        metrics=None,
        loss_weights=None,
    ):
        super().__init__(
            name=name,
            NumDomain=NumDomain,
            use_output_transform=use_output_transform,
            layer_size=layer_size,
            activation=activation,
            initializer=initializer,
            external_trainable_variables=external_trainable_variables,
            metrics=metrics,
            loss_weights=loss_weights,
        )

    @abstractmethod
    def gen_pde(self):
        pass

    @abstractmethod
    def gen_geomtime(self):
        pass

    @abstractmethod
    def gen_data(self):
        pass


class Lorenz_Inverse(InverseCase):
    def __init__(
        self,
        NumDomain=400,
        layer_size=[1] + [40] * 3 + [3],
        activation="tanh",
        initializer="Glorot uniform",
    ):
        self.C1 = dde.Variable(1.0)
        self.C2 = dde.Variable(1.0)
        self.C3 = dde.Variable(1.0)
        super().__init__(
            "Inverse problem for the Lorenz system",
            external_trainable_variables=[self.C1, self.C2, self.C3],
            NumDomain=NumDomain,
            use_output_transform=False,
            layer_size=layer_size,
            activation=activation,
            initializer=initializer,
        )

    def gen_testdata(self):
        import os

        basepath = os.path.abspath(__file__)
        folder = os.path.dirname(os.path.dirname(basepath))
        data_path = os.path.join(folder, "data/Lorenz.npz")
        data = np.load(data_path)
        return data["t"], data["y"]

    def gen_pde(self):
        def Lorenz_system(x, y):
            y1, y2, y3 = y[:, 0:1], y[:, 1:2], y[:, 2:]
            dy1_x = dde.grad.jacobian(y, x, i=0)
            dy2_x = dde.grad.jacobian(y, x, i=1)
            dy3_x = dde.grad.jacobian(y, x, i=2)
            return [
                dy1_x - self.C1 * (y2 - y1),
                dy2_x - y1 * (self.C2 - y3) + y2,
                dy3_x - y1 * y2 + self.C3 * y3,
            ]

        return Lorenz_system

    def gen_geomtime(self):
        return dde.geometry.TimeDomain(0, 3)

    def gen_data(self):
        ic1 = dde.icbc.IC(
            self.geomtime, lambda X: -8, lambda _, on_initial: on_initial, component=0
        )
        ic2 = dde.icbc.IC(
            self.geomtime, lambda X: 7, lambda _, on_initial: on_initial, component=1
        )
        ic3 = dde.icbc.IC(
            self.geomtime, lambda X: 27, lambda _, on_initial: on_initial, component=2
        )
        observe_t, ob_y = self.gen_testdata()
        observe_y0 = dde.icbc.PointSetBC(observe_t, ob_y[:, 0:1], component=0)
        observe_y1 = dde.icbc.PointSetBC(observe_t, ob_y[:, 1:2], component=1)
        observe_y2 = dde.icbc.PointSetBC(observe_t, ob_y[:, 2:3], component=2)
        return dde.data.PDE(
            self.geomtime,
            self.pde,
            [ic1, ic2, ic3, observe_y0, observe_y1, observe_y2],
            num_domain=self.NumDomain,
            num_boundary=2,
            anchors=observe_t,
        )

    def plot_result(self, solver, axes=None, exact=True):
        C1_true = 10
        C2_true = 15
        C3_true = 8 / 3

        C1_pred = bkd.to_numpy(self.C1)
        C2_pred = bkd.to_numpy(self.C2)
        C3_pred = bkd.to_numpy(self.C3)

        C1_error = np.abs(C1_true - C1_pred)
        C2_error = np.abs(C2_true - C2_pred)
        C3_error = np.abs(C3_true - C3_pred)

        print(f"C1 true: {C1_true}, C2 true: {C2_true}, C3 true: {C3_true}")
        print(f"C1 pred: {C1_pred}, C2 pred: {C2_pred}, C3 pred: {C3_pred}")
        print(f"C1 error: {C1_error}, C2 error: {C2_error}, C3 error: {C3_error}")

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


class Lorenz_Exogenous_Input_Inverse(InverseCase):
    def __init__(
        self,
        NumDomain=400,
        layer_size=[1] + [40] * 3 + [3],
        activation="tanh",
        initializer="Glorot uniform",
    ):
        self.C1 = dde.Variable(1.0)
        self.C2 = dde.Variable(1.0)
        self.C3 = dde.Variable(1.0)
        self.C1_true = 10
        self.C2_true = 15
        self.C3_true = 8 / 3
        super().__init__(
            "Inverse problem for the Lorenz system with exogenous input",
            external_trainable_variables=[self.C1, self.C2, self.C3],
            NumDomain=NumDomain,
            use_output_transform=False,
            layer_size=layer_size,
            activation=activation,
            initializer=initializer,
        )

    def gen_testdata(self):
        import scipy as sp
        from scipy.integrate import odeint

        def ex_func(t):
            spline = sp.interpolate.Rbf(
                time, ex_input, function="thin_plate", smooth=0, episilon=0
            )
            return spline(t)

        def LorezODE(x, t):
            x1, x2, x3 = x
            dxdt = [
                self.C1_true * (x2 - x1),
                x1 * (self.C2_true - x3) - x2,
                x1 * x2 - self.C3_true * x3 + ex_func(t),
            ]
            return dxdt

        maxtime = 3
        time = np.linspace(0, maxtime, 200)
        ex_input = 10 * np.sin(2 * np.pi * time)
        x0 = [-8, 7, 27]
        x = odeint(LorezODE, x0, time)
        return time[:, np.newaxis], x

    def gen_pde(self):
        def Lorenz_system(x, y, ex):
            y1, y2, y3 = y[:, 0:1], y[:, 1:2], y[:, 2:]
            dy1_x = dde.grad.jacobian(y, x, i=0)
            dy2_x = dde.grad.jacobian(y, x, i=1)
            dy3_x = dde.grad.jacobian(y, x, i=2)
            return [
                dy1_x - self.C1 * (y2 - y1),
                dy2_x - y1 * (self.C2 - y3) + y2,
                dy3_x - y1 * y2 + self.C3 * y3 - ex,
            ]

        return Lorenz_system

    def gen_geomtime(self):
        return dde.geometry.TimeDomain(0, 3)

    def gen_data(self):
        ic1 = dde.icbc.IC(
            self.geomtime, lambda X: -8, lambda _, on_initial: on_initial, component=0
        )
        ic2 = dde.icbc.IC(
            self.geomtime, lambda X: 7, lambda _, on_initial: on_initial, component=1
        )
        ic3 = dde.icbc.IC(
            self.geomtime, lambda X: 27, lambda _, on_initial: on_initial, component=2
        )
        observe_t, ob_y = self.gen_testdata()
        observe_y0 = dde.icbc.PointSetBC(observe_t, ob_y[:, 0:1], component=0)
        observe_y1 = dde.icbc.PointSetBC(observe_t, ob_y[:, 1:2], component=1)
        observe_y2 = dde.icbc.PointSetBC(observe_t, ob_y[:, 2:3], component=2)

        def ex_func2(t):
            import scipy as sp

            maxtime = 3
            time = np.linspace(0, maxtime, 200)
            ex_input = 10 * np.sin(2 * np.pi * time)
            spline = sp.interpolate.Rbf(
                time, ex_input, function="thin_plate", smooth=0, episilon=0
            )
            return spline(t[:, 0:])

        return dde.data.PDE(
            self.geomtime,
            self.pde,
            [ic1, ic2, ic3, observe_y0, observe_y1, observe_y2],
            num_domain=self.NumDomain,
            num_boundary=2,
            anchors=observe_t,
            auxiliary_var_function=ex_func2,
        )

    def plot_result(self, solver, axes=None, exact=True):
        C1_pred = bkd.to_numpy(self.C1)
        C2_pred = bkd.to_numpy(self.C2)
        C3_pred = bkd.to_numpy(self.C3)

        C1_error = np.abs(self.C1_true - C1_pred)
        C2_error = np.abs(self.C2_true - C2_pred)
        C3_error = np.abs(self.C3_true - C3_pred)

        print(
            f"C1 true: {self.C1_true}, C2 true: {self.C2_true}, C3 true: {self.C3_true}"
        )
        print(f"C1 pred: {C1_pred}, C2 pred: {C2_pred}, C3 pred: {C3_pred}")
        print(f"C1 error: {C1_error}, C2 error: {C2_error}, C3 error: {C3_error}")

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


class Brinkman_Forchheimer_Inverse(InverseCase):
    def __init__(
        self,
        NumDomain=100,
        layer_size=[1] + [20] * 3 + [1],
        activation="tanh",
        initializer="Glorot uniform",
    ):
        self.g = 1
        self.v = 1e-3
        self.e = 0.4
        self.v_e = dde.Variable(0.1)
        self.K = dde.Variable(0.1)
        self.v_e_true = 1e-3
        self.K_true = 1e-3
        super().__init__(
            "Inverse problem for Brinkman-Forchheimer model",
            external_trainable_variables=[self.v_e, self.K],
            NumDomain=NumDomain,
            use_output_transform=True,
            layer_size=layer_size,
            activation=activation,
            initializer=initializer,
            metrics=["l2 relative error"],
        )

    def sol(self, x):
        H = 1
        r = (self.v * self.e / (1e-3 * 1e-3)) ** (0.5)
        return (
            self.g * 1e-3 / self.v * (1 - np.cosh(r * (x - H / 2)) / np.cosh(r * H / 2))
        )

    def gen_pde(self):
        def pde(x, y):
            du_xx = dde.grad.hessian(y, x)
            return -self.v_e / self.e * du_xx + self.v * y / self.K - self.g

        return pde

    def gen_geomtime(self):
        return dde.geometry.Interval(0, 1)

    def gen_data(self):
        num = 5
        xvals = np.linspace(1 / (num + 1), 1, num, endpoint=False)
        yvals = self.sol(xvals)
        ob_x, ob_u = np.reshape(xvals, (-1, 1)), np.reshape(yvals, (-1, 1))
        observe_u = dde.icbc.PointSetBC(ob_x, ob_u, component=0)
        return dde.data.PDE(
            self.geomtime,
            self.pde,
            [observe_u],
            num_domain=self.NumDomain,
            num_boundary=0,
            num_test=500,
            solution=self.sol,
            train_distribution="uniform",
        )

    def output_transform(self, x, y):
        return x * (1 - x) * y

    def plot_result(self, solver, axes=None, exact=True):
        v_e_pred = bkd.to_numpy(self.v_e)
        K_pred = bkd.to_numpy(self.K)

        v_e_error = np.abs(self.v_e_true - v_e_pred)
        K_error = np.abs(self.K_true - K_pred)

        print(f"v_e true: {self.v_e_true}, K true: {self.K_true}")
        print(f"v_e pred: {v_e_pred}, K pred: {K_pred}")
        print(f"v_e error: {v_e_error}, K error: {K_error}")

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


class Diffusion_Inverse(InverseCase):
    def __init__(
        self,
        NumDomain=40,
        layer_size=[2] + [32] * 3 + [1],
        activation="tanh",
        initializer="Glorot uniform",
    ):
        self.C = dde.Variable(2.0)
        self.C_true = 1.0
        super().__init__(
            "Inverse problem for the diffusion equation",
            external_trainable_variables=self.C,
            NumDomain=NumDomain,
            use_output_transform=False,
            layer_size=layer_size,
            activation=activation,
            initializer=initializer,
            metrics=["l2 relative error"],
        )

    def sol(self, x):
        return np.sin(np.pi * x[:, 0:1]) * np.exp(-x[:, 1:])

    def gen_pde(self):
        def pde(x, y):
            dy_t = dde.grad.jacobian(y, x, i=0, j=1)
            dy_xx = dde.grad.hessian(y, x, i=0, j=0)
            return (
                dy_t
                - self.C * dy_xx
                + bkd.exp(-x[:, 1:])
                * (bkd.sin(np.pi * x[:, 0:1]) - np.pi**2 * bkd.sin(np.pi * x[:, 0:1]))
            )

        return pde

    def gen_geomtime(self):
        geom = dde.geometry.Interval(-1, 1)
        timedomain = dde.geometry.TimeDomain(0, 1)
        return dde.geometry.GeometryXTime(geom, timedomain)

    def gen_data(self):
        bc = dde.icbc.DirichletBC(
            self.geomtime, self.sol, lambda _, on_boundary: on_boundary
        )
        ic = dde.icbc.IC(self.geomtime, self.sol, lambda _, on_initial: on_initial)

        observe_x = np.vstack((np.linspace(-1, 1, num=10), np.full((10), 1))).T
        observe_y = dde.icbc.PointSetBC(observe_x, self.sol(observe_x), component=0)
        return dde.data.TimePDE(
            self.geomtime,
            self.pde,
            [bc, ic, observe_y],
            num_domain=self.NumDomain,
            num_boundary=20,
            num_initial=10,
            num_test=10000,
            anchors=observe_x,
            solution=self.sol,
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

    def plot_result(self, solver, colorbar=[0, 0, 0]):
        C_pred = bkd.to_numpy(self.C)

        C_error = np.abs(self.C_true - C_pred)

        print(f"v_e true: {self.C_true}")
        print(f"v_e pred: {C_pred}")
        print(f"v_e error: {C_error}")

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
