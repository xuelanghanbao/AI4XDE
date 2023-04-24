import torch
import numpy as np
from solver.PDESolver import PINNSolver

class RAR_G(PINNSolver):
    def __init__(self, case='Burgers', NumDomain=1000, epoch=100):
        super().__init__(name='RAR_G', NumDomain=NumDomain, epoch=epoch, case=case)

    def closure(self):
        X = self.geomtime.random_points(100000)
        Y = np.abs(self.model.predict(X, operator=self.pde))[:, 0]
        err_eq = torch.tensor(Y)
        X_ids = torch.topk(err_eq, self.NumDomain//100, dim=0)[1].cpu().numpy()
        self.data.add_anchors(X[X_ids])

        self.train_step(iteration=1000)

if __name__ == '__main__':
    solver = RAR_G(case='Burgers', NumDomain=2000//2, epoch=100)
    solver.train()
    solver.save(add_time=True)