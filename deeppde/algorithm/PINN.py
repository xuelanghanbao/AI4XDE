import torch
import numpy as np
from solver.PDESolver import PINNSolver

class PINN(PINNSolver):
    def __init__(self, case='Burgers', NumDomain=2000, epoch=1):
        super().__init__(name='PINN', case=case, NumDomain=NumDomain, epoch=epoch, )

if __name__ == '__main__':
    solver = PINN(case='Burgers', NumDomain=2000, epoch=1)
    solver.train()
    solver.save(add_time=True)