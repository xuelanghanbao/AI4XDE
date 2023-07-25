import numpy as np
import deepxde as dde
from ..utils import Visualization
from abc import ABC, abstractmethod


class FuncCases(ABC):
    def __init__(
        self,
        name,
        layer_size=[2] + [32] * 3 + [1],
        activation="tanh",
        initializer="Glorot uniform",
        metrics=None,
        loss_weights=None,
        external_trainable_variables=None,
    ):
        self.name = name
        self.metrics = metrics
        self.loss_weights = loss_weights
        self.external_trainable_variables = external_trainable_variables

        self.net = self.gen_net(layer_size, activation, initializer)
        self.data = self.gen_data()
        self.compile = self.gen_compile()
        self.testdata = None

    def gen_compile(self):
        def compile(
            model,
            optimizer,
            lr=None,
            loss="MSE",
            decay=None,
        ):
            model.compile(
                optimizer,
                lr,
                loss,
                self.metrics,
                decay,
                self.loss_weights,
                self.external_trainable_variables,
            )

        return compile

    def set_loss_weights(self, loss_weights):
        self.loss_weights = loss_weights
        self.compile = self.gen_compile()

    def gen_net(self, layer_size, activation, initializer):
        net = dde.maps.FNN(layer_size, activation, initializer)
        return net

    @abstractmethod
    def gen_data(self):
        pass

    @abstractmethod
    def gen_testdata(self):
        pass

    def get_testdata(self):
        if self.testdata is None:
            self.testdata = self.gen_testdata()
        return self.testdata

    def set_axes(self, axes):
        pass

    def plot_data(self, X, axes=None):
        pass

    def plot_result(self, solver, colorbar=None):
        pass


class FuncFromFormula(FuncCases):
    def __init__(
        self,
        NumTrain=16,
        NumTest=100,
        layer_size=[1] + [20] * 3 + [1],
        activation="tanh",
        initializer="Glorot uniform",
        metrics=["l2 relative error"],
    ):
        self.NumTrain = NumTrain
        self.NumTest = NumTest
        self.geomtime = self.gen_geomtime()
        super().__init__(
            name="FuncFromFormula",
            layer_size=layer_size,
            activation=activation,
            initializer=initializer,
            metrics=metrics,
        )

    def func(self, x):
        return x * np.sin(5 * x)

    def gen_geomtime(self):
        return dde.geometry.Interval(-1, 1)

    def gen_data(self):
        return dde.data.Function(self.geomtime, self.func, self.NumTrain, self.NumTest)

    def gen_testdata(self):
        x = self.geomtime.uniform_points(self.NumTest)
        y = self.func(x)
        return x, y

    def plot_result(self, solver, axes=None, exact=True):
        fig, axes = Visualization.plot_1D_result(self, solver, axes, exact, "x", "y")
        return fig, axes


class FuncFromData(FuncCases):
    def __init__(
        self,
        TrainData="./data/dataset.train",
        TestData="./data/dataset.test",
        layer_size=[1] + [50] * 3 + [1],
        activation="tanh",
        initializer="Glorot uniform",
        metrics=["l2 relative error"],
    ):
        import os

        basepath = os.path.abspath(__file__)
        folder = os.path.dirname(os.path.dirname(basepath))
        self.TrainData = os.path.join(folder, TrainData)
        self.TestData = os.path.join(folder, TestData)
        super().__init__(
            name="FuncFromData",
            layer_size=layer_size,
            activation=activation,
            initializer=initializer,
            metrics=metrics,
        )

    def gen_data(self):
        return dde.data.DataSet(
            fname_train=self.TrainData,
            fname_test=self.TestData,
            col_x=(0,),
            col_y=(1,),
            standardize=True,
        )

    def gen_testdata(self):
        [x, y] = self.data.test()
        return x, y

    def plot_result(self, solver, axes=None, exact=True):
        fig, axes = Visualization.plot_1D_result(self, solver, axes, exact, "x", "y")
        return fig, axes
