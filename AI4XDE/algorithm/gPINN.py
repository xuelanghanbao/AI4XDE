import deepxde as dde
from ..cases.PDECases import Burgers
from ..solver.PDESolver import PINNSolver


class gPINN(PINNSolver):
    def __init__(self, PDECase):
        self.add_gradient_enhanced_res(PDECase)
        super().__init__(name="PINN", PDECase=PDECase)

    def add_gradient_enhanced_res(self, PDECase):
        pde = PDECase.pde
        if dde.utils.get_num_args(pde) == 2:

            def g_pde(x, y):
                g_res = []
                res = pde(x, y)

                if isinstance(res, list):
                    g_res.extend(res)
                    res_dim = len(res)
                else:
                    g_res.append(res)
                    res_dim = 1
                    res = [res]

                x_dim = x.shape[1]
                for i in range(res_dim):
                    for j in range(x_dim):
                        res_ij = dde.grad.jacobian(res[i], x, j=j)
                        g_res.append(res_ij)

                return g_res

        elif dde.utils.get_num_args(pde) == 3:

            def g_pde(x, y, ex):
                g_res = []
                res = pde(x, y, ex)

                if isinstance(res, list):
                    g_res.extend(res)
                    res_dim = len(res)
                else:
                    g_res.append(res)
                    res_dim = 1
                    res = [res]

                x_dim = x.shape[1]
                for i in range(res_dim):
                    for j in range(x_dim):
                        res_ij = dde.grad.jacobian(res[i], x, j=j)
                        g_res.append(res_ij)

                return g_res

        PDECase.set_pde(g_pde)


if __name__ == "__main__":
    PDECase = Burgers(NumDomain=2000)
    solver = gPINN(PDECase=PDECase)
    solver.train()
    solver.save(add_time=True)
