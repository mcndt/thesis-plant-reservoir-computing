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

baseline_reservoirs = [
    ("env_all", ["input_Tac", "input_u", "input_hs", "input_Rg"],),
    ("env_temp", ["input_Tac"]),
    ("env_humidity", ["input_hs"]),
    ("env_PAR", ["input_Rg"]),
    ("env_wind", ["input_u"]),
]

heterogeneous_reservoirs = [
    ("state__het_all", (*state_variables,)),
]

input_names = {
    "input_Tac": "Air temperature",
    "input_hs": "Air humidity",
    "input_Rg": "$I_{{PAR}}$",
    "input_u": "Wind speed",
}
