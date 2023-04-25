import numpy as np
from solver.PDESolver import PINNSolver

class RAD(PINNSolver):
    def __init__(self, case, NumDomain, k=1, c=1):
        super().__init__(name=f'RAD_k_{k}_c_{c}', case=case, NumDomain=NumDomain)
        self.k = k
        self.c = c

    def closure(self):
        self.train_step()

        for i in range(100):
            X = self.geomtime.random_points(100000)
            Y = np.abs(self.model.predict(X, operator=self.pde)).astype(np.float64)
            err_eq = np.power(Y, self.k) / np.power(Y, self.k).mean() + self.c
            err_eq_normalized = (err_eq / sum(err_eq))[:, 0]
            X_ids = np.random.choice(a=len(X), size=self.NumDomain, replace=False, p=err_eq_normalized)
            X_selected = X[X_ids]
            self.data.replace_with_anchors(X_selected)

            self.train_step(iteration=1000)

if __name__ == '__main__':
    solver = RAD(case='Burgers', k=2, c=0, NumDomain=2000)
    solver.train()
    solver.save(add_time=True)