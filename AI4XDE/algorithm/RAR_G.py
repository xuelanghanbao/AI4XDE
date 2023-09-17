import numpy as np
import deepxde as dde
from ..cases.PDECases import AllenCahn
from ..solver.PDESolver import PINNSolver
import time


class RAR_G(PINNSolver):
    def __init__(self, PDECase, iter=100):
        dde.optimizers.config.set_LBFGS_options(maxiter=1000)
        self.iter = iter
        super().__init__(name="RAR_G", PDECase=PDECase)

    def closure(self):
        self.train_step()

        for i in range(self.iter):
            X = self.PDECase.geomtime.random_points(100000)
            Y = np.abs(self.model.predict(X, operator=self.PDECase.pde))[:, 0]
            t1 = time.time()
            X_ids = np.argpartition(Y, self.PDECase.NumDomain // 100, axis=0)
            print(f"{time.time()-t1} s")
            self.PDECase.data.add_anchors(X[X_ids])

            self.train_step(iterations=1000)


if __name__ == "__main__":
    PDECase = AllenCahn(NumDomain=2000 // 2)
    solver = RAR_G(PDECase=PDECase)
    solver.train()
    solver.save(add_time=True)
