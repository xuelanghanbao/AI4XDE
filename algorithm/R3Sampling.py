import numpy as np
import deepxde as dde
import deepxde.backend as bkd
from .cases.PDECases import Burgers
from .solver.PDESolver import PINNSolver

dde.optimizers.config.set_LBFGS_options(maxiter=1000)

class R3Sampling(PINNSolver):
    def __init__(self, 
                 PDECase, 
                 max_iter=100,
                 causally_sampling=False,
                 beta_lr = 0.001,
                 tol = 20):
        name = 'R3Sampling'
        if causally_sampling:
            name += '_causal'
            PDECase.data.pde = lambda x, y: PDECase.pde(x, y) * self.causal_weight(bkd.to_numpy(x), tensor=True)
        super().__init__(name=name, PDECase=PDECase)
        self.gamma = -0.5
        self.alpha = 5
        self.max_iter = max_iter
        self.causally_sampling = causally_sampling
        self.beta_lr = beta_lr
        self.tol = tol

    def update_gamma(self, loss):
        self.gamma += self.beta_lr * np.exp(-self.tol * loss)

    def causal_weight(self, X, tensor=False):
        X = X[:, -1]
        X = X / (max(X) - min(X))
        weight = (1-np.tanh(self.alpha*(X-self.gamma))) /2
        weight = weight.reshape(-1)
        if tensor:
            weight = bkd.pow(bkd.from_numpy(weight),1/2)
        return weight

    def residual(self, X, causally_sampling=False):
        res = np.abs(self.model.predict(X, operator=self.PDECase.pde))[:, 0]
        if causally_sampling:
            res = res * self.causal_weight(X)
            self.update_gamma(np.mean(res))
        return res

    def closure(self):
        self.train_step()

        for i in range(self.max_iter):

            X = self.PDECase.data.train_x_all
            res = self.residual(X, self.causally_sampling)
            threshold = np.mean(res)
            X_retained=X[np.where(res>threshold)]
            self.PDECase.data.replace_with_anchors(X_retained)

            N_r = len(X) - len(X_retained)
            if N_r > 0:
                X_resample = self.PDECase.geomtime.random_points(N_r)
                self.PDECase.data.add_anchors(X_resample)

            self.train_step(iterations=1000)

            

if __name__ == '__main__':
    PDECase = Burgers(NumDomain=2000)
    solver = R3Sampling(PDECase=PDECase)
    solver.train()
    solver.save(add_time=True)