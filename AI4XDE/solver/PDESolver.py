import os
import time
import numpy as np
import deepxde as dde

dde.config.set_default_float("float64")


class PINNSolver:
    def __init__(self, name, PDECase):
        self.name = name
        self.PDECase = PDECase

        self.model = dde.Model(PDECase.data, PDECase.net)
        self.error = []
        self.losshistory = None
        self.train_state = None
        self.train_cost = None

    def save(self, path=None, add_time=False):
        if path is None:
            path = f"./models/{self.name}_{self.PDECase.name}"
        if add_time:
            path += f'_{time.strftime("%Y%m%d_%H%M%S", time.localtime())}/'
        else:
            path += "/"

        if not os.path.exists(path):
            os.makedirs(path)
        self.model.save(path)
        dde.saveplot(
            self.losshistory,
            self.train_state,
            issave=True,
            isplot=False,
            output_dir=path,
        )
        np.savetxt(f"{path}{self.name}_error.txt", self.error)

    def plot_loss_history(self, axes=None, train=False, use_time=False):
        import matplotlib.pyplot as plt

        if axes is None:
            fig, axes = plt.subplots()
        else:
            fig = axes.get_figure()
        loss_train = np.sum(self.losshistory.loss_train, axis=1)
        loss_test = np.sum(self.losshistory.loss_test, axis=1)

        if use_time:
            x = (
                np.array(self.losshistory.steps)
                / self.losshistory.steps[-1]
                * self.train_cost
            )
            axes.set_xlabel("Time(s)")
        else:
            x = self.losshistory.steps
            axes.set_xlabel("Steps")

        if train:
            axes.semilogy(x, loss_train, label="Train loss")
            axes.semilogy(x, loss_test, label="Test loss")
        else:
            axes.semilogy(x, loss_test, label=self.name)

        axes.set_title(self.name)
        axes.set_ylabel("Loss")
        axes.legend()
        return fig, axes

    def eval(self):
        X, y = self.PDECase.gen_testdata()
        y_pred = self.model.predict(X)
        error = dde.metrics.l2_relative_error(y, y_pred)
        print("L2 relative error:", error)
        self.error.append(np.array(error))

    def train_step(self, lr=1e-3, iterations=15000, callbacks=None, eval=True):
        self.PDECase.compile(self.model, "adam", lr=lr)
        self.model.train(iterations=iterations, callbacks=callbacks)
        self.PDECase.compile(self.model, "L-BFGS")
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
