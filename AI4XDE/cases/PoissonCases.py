import numpy as np
import deepxde as dde
import deepxde.backend as bkd
from .PDECases import PDECases
from abc import abstractmethod
from ..utils.Visualization import *


class PoissonCase1D(PDECases):
    def __init__(
        self,
        name,
        NumDomain=2000,
        Interval=None,
        use_output_transform=False,
        layer_size=[2] + [32] * 3 + [1],
        activation="tanh",
        initializer="Glorot uniform",
        metrics=["l2 relative error"],
        loss_weights=None,
        external_trainable_variables=None,
    ):
        self.Interval = Interval
        super().__init__(
            name=name,
            NumDomain=NumDomain,
            use_output_transform=use_output_transform,
            layer_size=layer_size,
            activation=activation,
            initializer=initializer,
            metrics=metrics,
            loss_weights=loss_weights,
            external_trainable_variables=external_trainable_variables,
            visualization=Visualization_1D(x_label="x", y_label="y"),
        )

    @abstractmethod
    def func(self, x):
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

    def plot_result(self, solver, axes=None, exact=True):
        axes = self.Visualization.plot_line_1D(self, solver, exact, axes=axes)
        return axes


class Poisson_1D_Dirichlet(PoissonCase1D):
    def __init__(
        self,
        NumDomain=16,
        layer_size=[1] + [50] * 3 + [1],
        activation="tanh",
        initializer="Glorot uniform",
    ):
        self.Interval = [-1, 1]
        super().__init__(
            name="Poisson equation in 1D with Dirichlet boundary conditions",
            NumDomain=NumDomain,
            Interval=self.Interval,
            use_output_transform=False,
            layer_size=layer_size,
            activation=activation,
            initializer=initializer,
        )

    def func(self, x):
        return -np.pi**2 * bkd.sin(np.pi * x)

    def sol(self, x):
        return np.sin(np.pi * x)

    def gen_data(self):
        bc = dde.icbc.DirichletBC(
            self.geomtime, self.sol, lambda _, on_boundary: on_boundary
        )
        return dde.data.PDE(
            self.geomtime,
            self.pde,
            bc,
            self.NumDomain,
            2,
            solution=self.sol,
            num_test=100,
        )


class Poisson_1D_Dirichlet_Neumann(PoissonCase1D):
    def __init__(
        self,
        NumDomain=16,
        layer_size=[1] + [50] * 3 + [1],
        activation="tanh",
        initializer="Glorot uniform",
    ):
        self.Interval = [-1, 1]
        super().__init__(
            name="Poisson equation in 1D with Dirichlet/Neumann boundary conditions",
            NumDomain=NumDomain,
            Interval=self.Interval,
            use_output_transform=False,
            layer_size=layer_size,
            activation=activation,
            initializer=initializer,
        )

    def func(self, x):
        return 2

    def sol(self, x):
        return (x + 1) ** 2

    def gen_data(self):
        def boundary_l(x, on_boundary):
            return on_boundary and np.isclose(x[0], -1)

        def boundary_r(x, on_boundary):
            return on_boundary and np.isclose(x[0], 1)

        bc_l = dde.icbc.DirichletBC(self.geomtime, self.sol, boundary_l)
        bc_r = dde.icbc.NeumannBC(self.geomtime, lambda X: 2 * (X + 1), boundary_r)
        return dde.data.PDE(
            self.geomtime,
            self.pde,
            [bc_l, bc_r],
            self.NumDomain,
            2,
            solution=self.sol,
            num_test=100,
        )


class Poisson_1D_Dirichlet_Robin(PoissonCase1D):
    def __init__(
        self,
        NumDomain=16,
        layer_size=[1] + [50] * 3 + [1],
        activation="tanh",
        initializer="Glorot uniform",
    ):
        self.Interval = [-1, 1]
        super().__init__(
            name="Poisson equation in 1D with Dirichlet/Robin boundary conditions",
            NumDomain=NumDomain,
            Interval=self.Interval,
            use_output_transform=False,
            layer_size=layer_size,
            activation=activation,
            initializer=initializer,
        )

    def func(self, x):
        return 2

    def sol(self, x):
        return (x + 1) ** 2

    def gen_data(self):
        def boundary_l(x, on_boundary):
            return on_boundary and np.isclose(x[0], -1)

        def boundary_r(x, on_boundary):
            return on_boundary and np.isclose(x[0], 1)

        bc_l = dde.icbc.DirichletBC(self.geomtime, self.sol, boundary_l)
        bc_r = dde.icbc.RobinBC(self.geomtime, lambda X, y: y, boundary_r)
        return dde.data.PDE(
            self.geomtime,
            self.pde,
            [bc_l, bc_r],
            self.NumDomain,
            2,
            solution=self.sol,
            num_test=100,
        )


class Poisson_1D_Dirichlet_Periodic(PoissonCase1D):
    def __init__(
        self,
        NumDomain=16,
        layer_size=[1] + [50] * 3 + [1],
        activation="tanh",
        initializer="Glorot uniform",
    ):
        self.Interval = [-1, 1]
        super().__init__(
            name="Poisson equation in 1D with Dirichlet/Periodic boundary conditions",
            NumDomain=NumDomain,
            Interval=self.Interval,
            use_output_transform=False,
            layer_size=layer_size,
            activation=activation,
            initializer=initializer,
        )

    def func(self, x):
        return -np.pi**2 * bkd.sin(np.pi * x)

    def sol(self, x):
        return np.sin(np.pi * x)

    def gen_data(self):
        def boundary_l(x, on_boundary):
            return on_boundary and np.isclose(x[0], -1)

        def boundary_r(x, on_boundary):
            return on_boundary and np.isclose(x[0], 1)

        bc_l = dde.icbc.DirichletBC(self.geomtime, self.sol, boundary_l)
        bc2 = dde.icbc.PeriodicBC(self.geomtime, 0, boundary_r)
        return dde.data.PDE(
            self.geomtime,
            self.pde,
            [bc_l, bc2],
            self.NumDomain,
            2,
            solution=self.sol,
            num_test=100,
        )


class Poisson_1D_Dirichlet_PointSetOperator(PoissonCase1D):
    def __init__(
        self,
        NumDomain=16,
        layer_size=[1] + [50] * 3 + [1],
        activation="tanh",
        initializer="Glorot uniform",
    ):
        self.Interval = [-1, 1]
        super().__init__(
            name="Poisson equation in 1D with Dirichlet/PointSetOperator boundary conditions",
            NumDomain=NumDomain,
            Interval=self.Interval,
            use_output_transform=False,
            layer_size=layer_size,
            activation=activation,
            initializer=initializer,
        )

    def func(self, x):
        return 2

    def sol(self, x):
        return (x + 1) ** 2

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
        return dde.data.PDE(
            self.geomtime,
            self.pde,
            [bc_l, bc_r],
            self.NumDomain,
            2,
            solution=self.sol,
            num_test=100,
        )


class Poisson_1D_Hard_Boundary(PoissonCase1D):
    def __init__(
        self,
        NumDomain=64,
        layer_size=[1] + [50] * 3 + [1],
        activation="tanh",
        initializer="Glorot uniform",
    ):
        self.Interval = [0, np.pi]
        super().__init__(
            name="Poisson equation in 1D with hard boundary conditions",
            NumDomain=NumDomain,
            Interval=self.Interval,
            use_output_transform=True,
            layer_size=layer_size,
            activation=activation,
            initializer=initializer,
        )

    def func(self, x):
        summation = sum([i * bkd.sin(i * x) for i in range(1, 5)])
        return -summation - 8 * bkd.sin(8 * x)

    def sol(self, x):
        summation = sum([np.sin(i * x) / i for i in range(1, 5)])
        return x + summation + np.sin(8 * x) / 8

    def gen_data(self):
        return dde.data.PDE(
            self.geomtime, self.pde, [], self.NumDomain, solution=self.sol, num_test=400
        )

    def output_transform(self, x, y):
        return x * (np.pi - x) * y + x


class Poisson_1D_Fourier_Net(PoissonCase1D):
    def __init__(
        self,
        NumDomain=1280,
        layer_size=[1] + [100] * 3 + [1],
        activation="tanh",
        initializer="Glorot uniform",
    ):
        self.Interval = [0, 1]
        self.A = 2
        self.B = 50
        super().__init__(
            name="Poisson equation in 1D with Multi-scale Fourier feature networks",
            NumDomain=NumDomain,
            Interval=self.Interval,
            use_output_transform=True,
            layer_size=layer_size,
            activation=activation,
            initializer=initializer,
        )

    def gen_compile(
        self,
        metrics=None,
        loss_weights=None,
        external_trainable_variables=None,
    ):
        def compile(
            model,
            optimizer,
            lr=None,
            loss="MSE",
            decay=("inverse time", 2000, 0.9),
        ):
            model.compile(
                optimizer,
                lr,
                loss,
                metrics,
                decay,
                loss_weights,
                external_trainable_variables,
            )

        return compile

    def func(self, x):
        result = -((np.pi * self.A) ** 2) * bkd.sin(np.pi * self.A * x) - 0.1 * (
            np.pi * self.B
        ) ** 2 * bkd.sin(np.pi * self.B * x)
        return result

    def sol(self, x):
        return np.sin(np.pi * self.A * x) + 0.1 * np.sin(np.pi * self.B * x)

    def gen_net(self, layer_size, activation, initializer):
        return dde.nn.MsFFN(layer_size, activation, initializer, sigmas=[1, 10])

    def gen_data(self):
        bc = dde.icbc.DirichletBC(
            self.geomtime, self.sol, lambda _, on_boundary: on_boundary
        )
        return dde.data.PDE(
            self.geomtime,
            self.pde,
            bc,
            self.NumDomain,
            train_distribution="pseudo",
            solution=self.sol,
            num_test=10000,
        )


class Poisson_2D_L_Shaped(PDECases):
    def __init__(
        self,
        NumDomain=1200,
        layer_size=[2] + [50] * 4 + [1],
        activation="tanh",
        initializer="Glorot uniform",
    ):
        visualization = Visualization_2D(
            x_limit=[-1, 1], y_limit=[-1, 1], x_label="x1", y_label="x2"
        )
        super().__init__(
            name="Poisson equation over L-shaped domain",
            NumDomain=NumDomain,
            use_output_transform=False,
            layer_size=layer_size,
            activation=activation,
            initializer=initializer,
            metrics=None,
            visualization=visualization,
        )

    def gen_data(self):
        def boundary(_, on_boundary):
            return on_boundary

        bc = dde.icbc.DirichletBC(self.geomtime, lambda x: 0, boundary)
        return dde.data.PDE(
            self.geomtime,
            self.pde,
            bc,
            num_domain=self.NumDomain,
            num_boundary=120,
            num_test=1500,
        )

    def gen_geomtime(self):
        return dde.geometry.Polygon(
            [[0, 0], [1, 0], [1, -1], [-1, -1], [-1, 1], [0, 1]]
        )

    def gen_pde(self):
        def pde(x, y):
            dy_xx = dde.grad.hessian(y, x, i=0, j=0)
            dy_yy = dde.grad.hessian(y, x, i=1, j=1)
            return -dy_xx - dy_yy - 1

        return pde

    def gen_testdata(self):
        X = np.array(
            [[x, t] for x in np.linspace(-1, 1, 1000) for t in np.linspace(-1, 1, 1000)]
        )
        y = np.linspace(-1, 1, 1000).T
        return X, y

    def plot_result(self, solver, colorbar=None):
        from matplotlib import pyplot as plt

        X = np.array(
            [[x, t] for x in np.linspace(-1, 1, 1000) for t in np.linspace(-1, 1, 1000)]
        )
        # y = self.sol(X)
        model_y = solver.model.predict(X)
        model_y[self.geomtime.inside(X) == 0] = np.nan

        fig, axes = plt.subplots()
        ax = self.Visualization.plot_heatmap_2D(
            X, model_y, shape=[1000, 1000], axes=axes, title=solver.name
        )
        if colorbar:
            fig.colorbar(ax, ax=axes)
        plt.show()
        return fig, axes


class Poisson_1D_Unknown_Forcing_Field_Inverse(PoissonCase1D):
    def __init__(
        self,
        NumDomain=16,
        layer_size=[1, [20, 20], [20, 20], [20, 20], 2],
        activation="tanh",
        initializer="Glorot uniform",
    ):
        self.Interval = [-1, 1]
        super().__init__(
            name="Inverse problem for the Poisson equation with unknown forcing field",
            NumDomain=NumDomain,
            Interval=self.Interval,
            use_output_transform=False,
            layer_size=layer_size,
            activation=activation,
            initializer=initializer,
            loss_weights=[1, 100, 1000],
            metrics=None,
        )

    def gen_pde(self):
        def pde(x, y):
            u, q = y[:, 0:1], y[:, 1:2]
            du_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
            return -du_xx + q

        return pde

    def func(self, x):
        return -np.pi**2 * np.sin(np.pi * x)

    def sol(self, x):
        return np.sin(np.pi * x)

    def gen_data(self):
        bc = dde.icbc.DirichletBC(
            self.geomtime, self.sol, lambda _, on_boundary: on_boundary, component=0
        )
        num = 100
        ob_x = np.linspace(-1, 1, num).reshape(num, 1)
        ob_u = self.sol(ob_x)
        observe_u = dde.icbc.PointSetBC(ob_x, ob_u, component=0)
        return dde.data.PDE(
            self.geomtime,
            self.pde,
            [bc, observe_u],
            self.NumDomain,
            2,
            anchors=ob_x,
            num_test=1000,
        )

    def gen_net(self, layer_size, activation, initializer):
        return dde.nn.PFNN(layer_size, activation, initializer)


class Poisson_1D_Fractional_Inverse(PoissonCase1D):
    def __init__(
        self,
        NumDomain=20,
        layer_size=[1] + [20] * 4 + [1],
        activation="tanh",
        initializer="Glorot normal",
    ):
        self.alpha = dde.Variable(1.5)
        self.alpha_true = 1.8
        self.Interval = [-1, 1]
        super().__init__(
            name="Inverse problem for the fractional Poisson equation in 1D",
            NumDomain=NumDomain,
            Interval=self.Interval,
            external_trainable_variables=self.alpha,
            use_output_transform=True,
            layer_size=layer_size,
            activation=activation,
            initializer=initializer,
            # loss_weights=[1, 100],
            metrics=None,
        )

    def gen_pde(self):
        from scipy.special import gamma

        def fpde(x, y, int_mat):
            """(D_{0+}^alpha + D_{1-}^alpha) u(x)"""
            if isinstance(int_mat, (list, tuple)) and len(int_mat) == 3:
                int_mat = bkd.sparse_tensor(*int_mat)
                lhs = bkd.sparse_dense_matmul(int_mat, y)
            else:
                lhs = bkd.matmul(int_mat, y)
            lhs /= 2 * bkd.cos(self.alpha * np.pi / 2)
            rhs = gamma(bkd.to_numpy(self.alpha) + 2) * x
            return lhs - rhs[: bkd.size(lhs)]

        return fpde

    def func(self, x):
        pass

    def sol(self, x):
        return x * (np.abs(1 - x**2)) ** (self.alpha_true / 2)

    def gen_data(self):
        observe_x = np.linspace(-1, 1, num=20)[:, None]
        observe_y = dde.icbc.PointSetBC(observe_x, self.sol(observe_x))

        return dde.data.FPDE(
            self.geomtime,
            self.pde,
            self.alpha,
            observe_y,
            [101],
            meshtype="static",
            anchors=observe_x,
            solution=self.sol,
        )
        # L-BFGS optimizer is not supported for dynamic meshtype.
        # return dde.data.FPDE(
        #    self.geomtime,
        #    self.pde,
        #    self.alpha,
        #    observe_y,
        #    [100],
        #    meshtype="dynamic",
        #    num_domain=self.NumDomain,
        #    anchors=observe_x,
        #    solution=self.sol,
        #    num_test=100,
        # )

    def output_transform(self, x, y):
        return (1 - x**2) * y

    def plot_result(self, solver):
        alpha_pred = bkd.to_numpy(self.alpha)

        alpha_error = np.abs(alpha_pred - self.alpha_true)

        print(f"alpha true: {self.alpha_true}")
        print(f"alpha pred: {alpha_pred}")
        print(f"alpha error: {alpha_error}")
        super().plot_result(solver)


class Poisson_2D_Fractional_Inverse(PDECases):
    def __init__(
        self,
        NumDomain=64,
        layer_size=[2] + [20] * 4 + [1],
        activation="tanh",
        initializer="Glorot normal",
    ):
        self.alpha = dde.Variable(1.5)
        self.alpha_true = 1.8
        self.Interval = [-1, 1]
        visualization = Visualization_2D(
            x_limit=[-1, 1], y_limit=[-1, 1], x_label="x1", y_label="x2"
        )
        super().__init__(
            name="Inverse problem for the fractional Poisson equation in 2D",
            NumDomain=NumDomain,
            external_trainable_variables=self.alpha,
            use_output_transform=True,
            layer_size=layer_size,
            activation=activation,
            initializer=initializer,
            loss_weights=[1, 100],
            metrics=None,
            visualization=visualization,
        )

    def gen_pde(self):
        from scipy.special import gamma

        def fpde(x, y, int_mat):
            r"""\int_theta D_theta^alpha u(x)"""
            if isinstance(int_mat, (list, tuple)) and len(int_mat) == 3:
                int_mat = bkd.sparse_tensor(*int_mat)
                lhs = bkd.sparse_dense_matmul(int_mat, y)
            else:
                lhs = bkd.matmul(int_mat, y)
            lhs = lhs[:, 0]
            lhs *= -bkd.exp(
                bkd.lgamma((1 - self.alpha) / 2) + bkd.lgamma((2 + self.alpha) / 2)
            ) / (2 * np.pi**1.5)
            x = x[: bkd.size(lhs)]
            alpha = bkd.to_numpy(self.alpha)
            rhs = (
                2**alpha
                * gamma(2 + alpha / 2)
                * gamma(1 + alpha / 2)
                * (1 - (1 + alpha / 2) * bkd.sum(x**2, 1))
            )
            return lhs - rhs

        return fpde

    def gen_geomtime(self):
        return dde.geometry.Disk([0, 0], 1)

    def sol(self, x):
        return (1 - np.linalg.norm(x, axis=1, keepdims=True) ** 2) ** (
            1 + self.alpha_true / 2
        )

    def gen_data(self):
        observe_x = self.geomtime.random_points(30)
        observe_y = dde.icbc.PointSetBC(observe_x, self.sol(observe_x))

        # L-BFGS optimizer is not supported for dynamic meshtype.
        return dde.data.FPDE(
            self.geomtime,
            self.pde,
            self.alpha,
            observe_y,
            [8, 100],
            num_domain=self.NumDomain,
            anchors=observe_x,
            solution=self.sol,
        )

    def output_transform(self, x, y):
        return (1 - bkd.sum(x**2, 1, keepdims=True)) * y

    def plot_result(self, solver, colorbar=[0, 0, 0]):
        alpha_pred = bkd.to_numpy(self.alpha)

        alpha_error = np.abs(alpha_pred - self.alpha_true)

        print(f"alpha true: {self.alpha_true}")
        print(f"alpha pred: {alpha_pred}")
        print(f"alpha error: {alpha_error}")

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
        shape = [1000, 1000]
        axs.append(
            self.Visualization.plot_heatmap_2D(
                X, y, shape, axes=axes[0], title="Exact solution"
            )
        )
        axs.append(
            self.Visualization.plot_heatmap_2D(
                X, model_y, shape, axes=axes[1], title=solver.name
            )
        )
        axs.append(
            self.Visualization.plot_heatmap_2D(
                X, np.abs(model_y - y), shape, axes=axes[2], title="Absolute error"
            )
        )

        for needColorbar, ax, axe in zip(colorbar, axs, axes):
            if needColorbar:
                fig.colorbar(ax, ax=axe)
        plt.show()
        return fig, axes
