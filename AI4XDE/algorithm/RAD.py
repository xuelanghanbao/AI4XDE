import numpy as np
import deepxde as dde
from ..cases.PDECases import Burgers
from ..solver.PDESolver import PINNSolver

dde.optimizers.config.set_LBFGS_options(maxiter=1000)


class RAD(PINNSolver):
    def __init__(self, PDECase, iter=100, k=1, c=1):
        super().__init__(name=f"RAD_k_{k}_c_{c}", PDECase=PDECase)
        self.iter = iter
        self.k = k
        self.c = c

    def closure(self):
        self.train_step()

        for i in range(self.iter):
            sample_num = self.PDECase.NumDomain * 10
            X = self.PDECase.geomtime.random_points(sample_num)
            Y = np.abs(self.model.predict(X, operator=self.PDECase.pde)).astype(
                np.float64
            )
            if Y.shape[0] != sample_num:
                Y = np.sum(Y, axis=0)
            err_eq = np.power(Y, self.k) / np.power(Y, self.k).mean() + self.c
            err_eq_normalized = (err_eq / sum(err_eq))[:, 0]
            X_ids = np.random.choice(
                a=len(X),
                size=self.PDECase.NumDomain,
                replace=False,
                p=err_eq_normalized,
            )
            X_selected = X[X_ids]
            self.PDECase.data.replace_with_anchors(X_selected)

            self.train_step(iterations=1000)


if __name__ == "__main__":
    PDECase = Burgers(NumDomain=2000)
    solver = RAD(PDECase=PDECase, k=1, c=1)
    solver.train()
    solver.save(add_time=True)
