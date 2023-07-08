__all__ = [
    "FuncCases",
    "FuncFromFormula",
    "FuncFromData",
    "PoissonCases",
    "Poisson_1D_Dirichlet",
    "Poisson_1D_Dirichlet_Neumann",
    "Poisson_1D_Dirichlet_Robin",
    "Poisson_1D_Dirichlet_Periodic",
    "Poisson_1D_Dirichlet_PointSetOperator",
    "Poisson_1D_Hard_Boundary",
    "Poisson_1D_Fourier_Net",
    "Poisson_2D_L_Shaped",
    "PDECases",
    "Wave",
    "Diffusion_Reaction_Inverse",
    "A_Simple_ODE",
    "LotkaVolterra",
    "SecondOrderODE",
    "Laplace_disk",
    "Helmholtz",
    "Helmholtz_Hole",
    "Helmholtz_Sound_hard_Absorbing",
    "Kovasznay_Flow",
    "Euler_Beam",
    "Burgers",
    "Heat",
    "Diffusion",
    "Diffusion_reaction",
    "AllenCahn",
    "Klein_Gordon",
    "IDE",
    "Volterra_IDE",
    "Fractional_Poisson_1D",
    "Fractional_Poisson_2D",
    "Fractional_Poisson_3D",
    "Fractional_Diffusion_1D"
]

from .PDECases import Burgers, Heat, AllenCahn, Diffusion, Klein_Gordon, Wave, Diffusion_Reaction_Inverse, A_Simple_ODE, LotkaVolterra, SecondOrderODE, Laplace_disk, Helmholtz, Helmholtz_Hole, Helmholtz_Sound_hard_Absorbing, Kovasznay_Flow, Euler_Beam, Diffusion_reaction, IDE, Volterra_IDE, Fractional_Poisson_1D, Fractional_Poisson_2D,Fractional_Poisson_3D,Fractional_Diffusion_1D
from .FuncCases import FuncCases, FuncFromFormula, FuncFromData
from .PoissonCases import Poisson_1D_Dirichlet, Poisson_1D_Dirichlet_Neumann, Poisson_1D_Dirichlet_Robin, Poisson_1D_Dirichlet_Periodic, Poisson_1D_Dirichlet_PointSetOperator, Poisson_1D_Hard_Boundary, Poisson_1D_Fourier_Net, Poisson_2D_L_Shaped