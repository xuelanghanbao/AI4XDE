import numpy as np
import deepxde as dde
import deepxde.backend as bkd
from ..solver.PDESolver import PINNSolver


class Domain_decomposition:
    def __init__(self, PDECase, segment):
        self.PDECase = PDECase
        self.segment = segment
        self.window_func_list = []
        self.pdeCase_list = self.gen_pdeCase_list()

    def gen_pdeCase_list(self):
        domain_list = self.domain_decomposition()
        pdeCase_list = []
        for domain in domain_list:
            pdeCase_list.append(self.gen_pde(domain))
        return pdeCase_list

    def gen_pde(self, domain):
        pdecase_class = self.PDECase.__class__
        window_func = self.get_window_func([domain.l, domain.r])
        self.window_func_list.append(window_func)

        class PDE(pdecase_class):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)

            def gen_geomtime(self):
                return domain

            def output_transform(self, x, y):
                y_parent = super().output_transform(x, y)
                if y_parent is not None:
                    y = y_parent
                y = window_func(x) * y
                return y

        pde = PDE()
        return pde

    def get_window_func(self, x_limit):
        def window_func(x):
            x = bkd.sigmoid(x - x_limit[0]) * bkd.sigmoid(x_limit[1] - x)
            return x

        return window_func

    def domain_decomposition(self):
        geomtime = self.PDECase.geomtime
        if isinstance(geomtime, dde.geometry.Interval):
            x_limit = [geomtime.l, geomtime.r]
            x_list = np.linspace(x_limit[0], x_limit[1], self.segment + 1)
            domain_list = []
            for i in range(self.segment):
                domain_list.append(dde.geometry.Interval(x_list[i], x_list[i + 1]))
        else:
            # TODO: support other geomtime
            raise ValueError("Only support geomtime Interval")
        return domain_list

    def plot_window_func(self):
        axes = self.PDECase.Visualization.set_axes_1D(title=self.PDECase.name)
        X, y = self.PDECase.get_testdata()
        for i, window_func in enumerate(self.window_func_list):
            X_tensor = bkd.from_numpy(X)
            window_fun = window_func(X_tensor)
            window_fun = bkd.to_numpy(window_fun)
            axes.plot(X, window_fun, "--", label=f"Prediction{i}")
        axes.legend()

    def plot_subdomains(self):
        axes = self.PDECase.Visualization.set_axes_1D(title=self.PDECase.name)
        for i, pde in enumerate(self.pdeCase_list):
            x = np.linspace(pde.geomtime.l, pde.geomtime.r, 100)
            axes.plot(x, np.zeros_like(x), label=f"subdomain {i}")
        axes.legend()


class FBPINN(PINNSolver):
    def __init__(self, PDECase, segment):
        self.domains = Domain_decomposition(PDECase, segment)
        super().__init__(name="FBPINN", PDECase=PDECase)
        self.model_list = [
            dde.Model(pde.data, pde.net) for pde in self.domains.pdeCase_list
        ]

    def train_step(self, lr=1e-3, iterations=1000, callbacks=None, eval=True):
        for pde, model in zip(self.domains.pdeCase_list, self.model_list):
            self.compile(pde, model, "adam", lr=lr)
            model.train(iterations=iterations, callbacks=callbacks)
            self.compile(pde, model, "L-BFGS")
            self.losshistory, self.train_state = model.train()
            if eval:
                self.eval()

    def closure(self):
        self.train_step()

    def compile(
        self,
        pde,
        model,
        optimizer,
        lr=None,
        loss="MSE",
        metrics=None,
        decay=None,
        loss_weights=None,
        external_trainable_variables=None,
    ):
        import torch
        from deepxde import gradients as grad
        from deepxde import losses as losses_module

        loss_fn = losses_module.get(loss)

        def outputs_losses(training, inputs, targets, auxiliary_vars, losses_fn):
            model.net.auxiliary_vars = auxiliary_vars
            model.net.train(mode=training)
            if isinstance(inputs, tuple):
                inputs = tuple(
                    map(lambda x: torch.as_tensor(x).requires_grad_(), inputs)
                )
            else:
                inputs = torch.as_tensor(inputs)
                inputs.requires_grad_()
            outputs_ = self.get_predict(inputs)
            # Data losses
            if targets is not None:
                targets = torch.as_tensor(targets)
            losses = losses_fn(targets, outputs_, loss_fn, inputs, model)
            if not isinstance(losses, list):
                losses = [losses]
            losses = torch.stack(losses)
            # Weighted losses
            if loss_weights is not None:
                losses *= torch.as_tensor(loss_weights)
            # Clear cached Jacobians and Hessians.
            grad.clear()
            return outputs_, losses

        def outputs_losses_train(inputs, targets, auxiliary_vars):
            return outputs_losses(
                True, inputs, targets, auxiliary_vars, model.data.losses_train
            )

        def outputs_losses_test(inputs, targets, auxiliary_vars):
            return outputs_losses(
                False, inputs, targets, auxiliary_vars, model.data.losses_test
            )

        def train_step(inputs, targets, auxiliary_vars):
            def closure():
                losses = outputs_losses_train(inputs, targets, auxiliary_vars)[1]
                total_loss = torch.sum(losses)
                model.opt.zero_grad()
                total_loss.backward()
                return total_loss

            model.opt.step(closure)
            if model.lr_scheduler is not None:
                model.lr_scheduler.step()

        # Callables
        pde.compile(model, optimizer, lr, loss, decay)
        model.outputs_losses_train = outputs_losses_train
        model.outputs_losses_test = outputs_losses_test
        model.train_step = train_step

    def get_predict(self, input):
        result = None
        for model in self.model_list:
            y = model.net(input)
            if result is None:
                result = y
            else:
                result += y
        return result

    def eval(self):
        X, y = self.PDECase.get_testdata()
        X_tensor = bkd.from_numpy(X)
        y_pred = self.get_predict(X_tensor)
        y_pred = bkd.to_numpy(y_pred)
        error = dde.metrics.l2_relative_error(y, y_pred)
        print("L2 relative error:", error)
        self.error.append(np.array(error))


if __name__ == "__main__":
    from ..cases.PDECases import Euler_Beam

    PDECase = Euler_Beam(
        NumDomain=10,
        layer_size=[1] + [10] * 2 + [1],
    )
    solver = FBPINN(PDECase=PDECase, segment=3)
    solver.train()
    solver.save(add_time=True)
