from ..cases.PDECases import Burgers
from ..solver.PDESolver import PINNSolver


class PINN(PINNSolver):
    """Implementation of PINN algorithm.
    Implementation of PINN algorithm in paper https://linkinghub.elsevier.com/retrieve/pii/S0021999118307125.
    """

    def __init__(self, PDECase):
        super().__init__(name="PINN", PDECase=PDECase)


if __name__ == "__main__":
    PDECase = Burgers(NumDomain=2000)
    solver = PINN(PDECase=PDECase)
    solver.train()
    solver.save(add_time=True)
