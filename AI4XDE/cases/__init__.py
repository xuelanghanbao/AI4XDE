__all__ = [
    "PDECases",
    "Burgers",
    "AllenCahn",
    "Diffusion",
    "Wave",
    "Diffusion_Reaction_Inverse",
    "A_Simple_ODE",
    "LotkaVolterra",
    "FuncCases",
    "FuncFromFormula",
    "FuncFromData"
]

from .PDECases import Burgers, AllenCahn, Diffusion, Wave, Diffusion_Reaction_Inverse, A_Simple_ODE, LotkaVolterra
from .FuncCases import FuncCases, FuncFromFormula, FuncFromData