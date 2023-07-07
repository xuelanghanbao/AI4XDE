__all__ = [
    "PDECases",
    "Burgers",
    "AllenCahn",
    "Diffusion",
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
    "FuncCases",
    "FuncFromFormula",
    "FuncFromData"
]

from .PDECases import Burgers, AllenCahn, Diffusion, Wave, Diffusion_Reaction_Inverse, A_Simple_ODE, LotkaVolterra, SecondOrderODE, Laplace_disk, Helmholtz, Helmholtz_Hole, Helmholtz_Sound_hard_Absorbing, Kovasznay_Flow
from .FuncCases import FuncCases, FuncFromFormula, FuncFromData