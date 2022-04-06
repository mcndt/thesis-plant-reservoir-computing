"""
This file contains constants defining which inputs, outputs and state variables 
of the HydroShoot dataset are to be used for reservoir computing experiments.
"""

input_targets = [
    "input_Tac",
    "input_u",
    "input_hs",
    "input_Rg",
]

output_targets = [
    "output_Rg",
    "output_An",
    "output_E",
    "output_Tleaf",
]

targets = [*input_targets, *output_targets]

state_variables = [
    "state_An",
    "state_E",
    "state_Eabs",
    "state_Ei",
    "state_Flux",
    "state_FluxC",
    "state_Tlc",
    "state_gb",
    "state_gs",
    "state_psi_head",
    "state_u",
]

