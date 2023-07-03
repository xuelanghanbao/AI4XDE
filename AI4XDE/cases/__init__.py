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
    "FuncCases",
    "FuncFromFormula",
    "FuncFromData"
]

from .PDECases import Burgers, AllenCahn, Diffusion, Wave, Diffusion_Reaction_Inverse, A_Simple_ODE, LotkaVolterra, SecondOrderODE
from .FuncCases import FuncCases, FuncFromFormula, FuncFromData