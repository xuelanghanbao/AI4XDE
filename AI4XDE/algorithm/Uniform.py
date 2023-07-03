import skopt
import numpy as np
import deepxde as dde
from ..cases.PDECases import Burgers
from ..solver.PDESolver import PINNSolver
from distutils.version import LooseVersion

class Uniform(PINNSolver):
    def __init__(self, PDECase, method):
        self.method = method
        super().__init__(name=f'Uniform_{method}', PDECase=PDECase)
    
    def gen_data(self):
        if self.method == 'Grid':
            data = dde.data.TimePDE(self.PDECase.geomtime, self.PDECase.pde, [], num_domain=self.PDECase.NumDomain, train_distribution='uniform')
        elif self.method == 'Random':
            data = dde.data.TimePDE(self.PDECase.geomtime, self.PDECase.pde, [], num_domain=self.PDECase.NumDomain, train_distribution='pseudo')
        elif self.method in ['LHS', 'Halton', 'Hammersley', 'Sobol']:
            sample_pts = self.quasirandom(self.PDECase.NumDomain, self.method)
            data = dde.data.TimePDE(self.PDECase.geomtime, self.PDECase.pde, [], num_domain=0, train_distribution='uniform', anchors=sample_pts)
        return data
    
    def quasirandom(self, n_samples, sampler):
        space = [(-1.0, 1.0), (0.0, 1.0)]
        if sampler == "LHS":
            sampler = skopt.sampler.Lhs(
                lhs_type="centered", criterion="maximin", iterations=1000
            )
        elif sampler == "Halton":
            sampler = skopt.sampler.Halton(min_skip=-1, max_skip=-1)
        elif sampler == "Hammersley":
            sampler = skopt.sampler.Hammersly(min_skip=-1, max_skip=-1)
        elif sampler == "Sobol":
            # Remove the first point [0, 0, ...] and the second point [0.5, 0.5, ...], which
            # are too special and may cause some error.
            if LooseVersion(skopt.__version__) < LooseVersion("0.9"):
                sampler = skopt.sampler.Sobol(min_skip=2, max_skip=2, randomize=False)
            else:
                sampler = skopt.sampler.Sobol(skip=0, randomize=False)
                return np.array(
                    sampler.generate(space, n_samples + 2)[2:]
                )
        return np.array(sampler.generate(space, n_samples))
    
if __name__ == '__main__':
    PDECase = Burgers(NumDomain=2000)
    solver = Uniform(PDECase=PDECase, method='Grid')

    #solver = Uniform(PDECase=PDECase, method='Random')
    #solver = Uniform(PDECase=PDECase, method='LHS')
    #solver = Uniform(PDECase=PDECase, method='Halton')
    #solver = Uniform(PDECase=PDECase, method='Hammersley')
    #solver = Uniform(PDECase=PDECase, method='Sobol')

    solver.train()
    solver.save(add_time=True)
    solver.plot_loss_history(train=True, use_time=True)