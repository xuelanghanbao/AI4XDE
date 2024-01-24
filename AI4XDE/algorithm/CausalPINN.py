import numpy as np
import deepxde as dde
import deepxde.backend as bkd
from ..cases.PDECases import AllenCahn
from ..solver.PDESolver import PINNSolver


class CausalPINN(PINNSolver):
    def __init__(
        self,
        PDECase,
        t_num=10,
        num_per_t=200,
        iter=10,
        warmup_iter=1000,
        k=1,
        c=0,
    ):
        self.t_num = t_num
        self.num_per_t = num_per_t
        self.iter = iter
        self.warmup_iter = warmup_iter
        self.k = k
        self.c = c
        self.weight = None
        dde.optimizers.config.set_LBFGS_options(maxiter=1000)
        assert isinstance(
            PDECase.geomtime, dde.geometry.GeometryXTime
        ), "CausalPINN only support GeometryXTime"
        super().__init__(name="CausalPINN", PDECase=PDECase)

    def get_weight_func(self, weights, t_list):
        def weight_func(x):
            t = x[:, -1]
            x_weight = bkd.zeros_like(t)
            for i in range(len(t_list) - 1):
                x_weight[(t >= t_list[i]) & (t < t_list[i + 1])] = weights[i]

            return x_weight.reshape(-1, 1)

        return weight_func

    def get_pde_error(self, x):
        Y = np.abs(self.model.predict(x, operator=self.PDECase.pde)).astype(np.float64)
        if Y.shape[0] != len(x):
            Y = np.sum(Y, axis=0).squeeze()

        Y = Y.squeeze()
        return Y

    def set_pde_weight(self, weight_func):
        pde = self.PDECase.pde
        pde_num_args = dde.utils.get_num_args(pde)

        def w_res(x, res):
            weight_func_x = weight_func(x)
            if isinstance(res, list):
                res = [weight_func_x * r for r in res]
            else:
                res = weight_func_x * res

            return res

        if pde_num_args == 2:

            def weight_pde(x, y):
                res = pde(x, y)
                return w_res(x, res)

        elif pde_num_args == 3:

            def weight_pde(x, y, ex):
                res = pde(x, y, ex)
                return w_res(x, res)

        self.PDECase.set_pde(weight_pde)
        self.model.data = self.PDECase.data

    def update_weights(self):
        sample_num = self.num_per_t

        t_list = np.linspace(
            self.PDECase.geomtime.timedomain.t0,
            self.PDECase.geomtime.timedomain.t1,
            self.t_num + 1,
        )
        X_in_t = None
        for t0, t1 in zip(t_list[:-1], t_list[1:]):
            timedomain = dde.geometry.TimeDomain(t0, t1)
            geomtime = dde.geometry.GeometryXTime(
                self.PDECase.geomtime.geometry, timedomain
            )
            if X_in_t is None:
                X_in_t = geomtime.uniform_points(sample_num)
            else:
                X_in_t = np.concatenate(
                    (X_in_t, geomtime.uniform_points(sample_num)), axis=0
                )

        Y = self.get_pde_error(X_in_t)
        err_eq = np.power(Y, self.k) / np.power(Y, self.k).mean() + self.c
        err_eq_normalized = err_eq / sum(err_eq)

        err_per_t = []
        for i in range(self.t_num):
            err_per_t.append(
                err_eq_normalized[i * sample_num : (i + 1) * sample_num].mean()
            )
        err_per_t = np.array(err_per_t)

        weights = np.zeros_like(err_per_t)
        epsilon = 100
        for i in range(self.t_num):
            if i == 0:
                weights[i] = 1
            else:
                weights[i] = np.exp(-epsilon * err_per_t[: i - 1].sum())
        self.weight = weights
        weight_func = self.get_weight_func(weights, t_list)
        self.set_pde_weight(weight_func)

    def plot_weights(self):
        import matplotlib.pyplot as plt

        t_list = np.linspace(
            self.PDECase.geomtime.timedomain.t0,
            self.PDECase.geomtime.timedomain.t1,
            self.t_num + 1,
        )
        weights = self.weight
        fig, axes = plt.subplots()
        axes.scatter(t_list[:-1], weights)
        axes.set_xlabel("t")
        axes.set_ylabel("weights")
        plt.show()

    def closure(self, plot=False):
        self.train_step(iterations=self.warmup_iter)

        for i in range(self.iter):
            self.update_weights()
            if plot:
                self.plot_weights()
            self.train_step(iterations=1000)


if __name__ == "__main__":
    PDECase = AllenCahn(NumDomain=2000)
    solver = CausalPINN(PDECase=PDECase)
    solver.train(plot=True)
    solver.save(add_time=True)
