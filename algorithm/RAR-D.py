import numpy as np
import deepxde as dde
from cases.PDECases import Burgers
from solver.PDESolver import PINNSolver

dde.optimizers.config.set_LBFGS_options(maxiter=1000)

class RAR_D(PINNSolver):
    def __init__(self, PDECase, k=2, c=0):
        super().__init__(name=f'RAR_D_k_{k}_c_{c}', PDECase=PDECase)
        self.name = f'RAR_D_k_{k}_c_{c}'
        self.k = k
        self.c = c
    
    def closure(self):
        self.train_step()

        for i in range(100):
            X = self.PDECase.geomtime.random_points(100000)
            Y = np.abs(self.model.predict(X, operator=self.PDECase.pde)).astype(np.float64)
            err_eq = np.power(Y, self.k) / np.power(Y, self.k).mean() + self.c
            err_eq_normalized = (err_eq / sum(err_eq))[:, 0]
            X_ids = np.random.choice(a=len(X), size=self.PDECase.NumDomain//100, replace=False, p=err_eq_normalized)
            self.PDECase.data.add_anchors(X[X_ids])

            self.train_step(iterations=1000)
    
if __name__ == '__main__':
    PDECase = Burgers(NumDomain=2000//2)
    solver = RAR_D(PDECase=PDECase, k=2, c=0)
    solver.train()
    solver.save(add_time=True)
    PDECase.plot_heatmap(solver)