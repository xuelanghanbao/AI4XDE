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
    ):
        super().__init__(
            name=name,
            NumDomain=NumDomain,
            use_output_transform=use_output_transform,
            layer_size=layer_size,
            activation=activation,
            initializer=initializer,
            external_trainable_variables=external_trainable_variables,
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

    def plot_result(self, solver):
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
