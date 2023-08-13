import numpy as np
import deepxde as dde
from ..utils import Math
from ..cases.PDECases import Burgers
from ..solver.PDESolver import PINNSolver


class FI_PINN(PINNSolver):
    def __init__(
        self,
        PDECase,
        max_iter=10,
        epsilon_r=0.1,
        epsilon_p=0.1,
        max_iter_SAIS=100,
        N_1=300,
        N_2=1000,
        p0=0.1,
    ):
        self.max_iter = max_iter
        self.epsilon_r = epsilon_r
        self.epsilon_p = epsilon_p
        self.max_iter_SAIS = max_iter_SAIS
        self.N_1 = N_1
        self.N_2 = N_2
        self.p0 = p0
        super().__init__(name="FI-PINN", PDECase=PDECase)

    def g(self, X):
        Y = np.abs(self.model.predict(X, operator=self.PDECase.pde)).astype(np.float64)
        if Y.shape[0] != X.shape[0]:
            Y = np.sum(Y, axis=0)
        return Y - self.epsilon_r

    def random_points(self, N, mul, sigma, sample_rate=10):
        result = []
        while True:
            X = self.PDECase.geomtime.random_points(N * sample_rate)
            X = Math.transform_uniform_to_normal_2D(X)
            X = Math.transform_normal_to_truncated_normal_on_geomtime(
                X, self.PDECase.geomtime, mul, sigma
            )
            if X.shape[0] >= N:
                result = X[0:N, :]
                break
            else:
                result.append(X)
                N -= X.shape[0]
        return np.array(result)

    def get_probility(self, X, loc, scale):
        from scipy import stats

        y = stats.norm.pdf(X, loc=loc, scale=scale)
        return y

    def SAIS(self):
        mul_i = np.array([0.0, 0.0])
        sigma_i = np.array([1.0])
        for i in range(self.max_iter_SAIS):
            h = self.random_points(self.N_1, mul_i, sigma_i)
            h_g = self.g(h).squeeze()
            index = np.argsort(-h_g)
            h_g = h_g[index]
            h = h[index, :]
            N_eta = h_g[0]
            N_p = int(np.floor(self.p0 * self.N_1))
            h = h[0:N_p, :]
            if N_eta <= N_p:
                h_p = self.get_probility(h, np.array([0.0, 0.0]), np.array([1.0]))
                mul_i = np.sum(h * h_p) / np.sum(h_p)
                sigma_i = np.sum((h - mul_i) ** 2) / (N_p - 1)
            else:
                break
        mul_opt = np.mean(h[0:N_p, :], axis=0)
        sigma_opt = np.var(h[0:N_p, :], axis=0, ddof=1)

        N2_sample = self.random_points(self.N_2, mul_opt, sigma_opt)

        Pf = (
            np.sum(
                (self.g(N2_sample) > 0)
                * self.get_probility(N2_sample, np.array([0.0, 0.0]), np.array([1.0]))
                / self.get_probility(N2_sample, mul_i, sigma_i)
            )
            / self.N_2
        )

        N2_g = self.g(N2_sample).squeeze()
        # if np.mean(N2_g) < self.epsilon_r:
        # self.epsilon_r = np.mean(N2_g)
        index = np.where(N2_g > 0)
        N2_g = N2_g[index]
        D_adaptive = N2_sample[index]

        return Pf, D_adaptive

    def closure(self):
        self.train_step()

        for i in range(self.max_iter):
            Pf, D_adaptive = self.SAIS()
            if Pf < self.epsilon_p:
                break
            if D_adaptive.shape[0] != 0:
                print(f"iter:{i}, add {D_adaptive.shape[0]} samples")
                self.PDECase.data.add_anchors(D_adaptive)
                self.PDECase.Visualization.plot_data_2D(D_adaptive)
            else:
                print("D_adaptive.shape[0] = 0:")
            self.train_step(iterations=1000, lr=0.0001)


if __name__ == "__main__":
    PDECase = Burgers(NumDomain=2000)
    solver = FI_PINN(PDECase=PDECase)
    solver.train()
    solver.save(add_time=True)
