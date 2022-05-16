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

state_variables = [
    "state_An",
    "state_Tlc",
    "state_Flux",
    "state_gs",
    "state_psi_head",
    "state_E",
    # "state_gb",
    # "state_FluxC",
    # "state_Eabs",
    # "state_Ei",
    # "state_u",
]

measurable_reservoirs = [
    "state_An",
    "state_Tlc",
    "state_Flux",
    "state_gs",
    "state_psi_head",
    "state_E",
    # "state_gb",
]

targets = [*input_targets, *output_targets]


baseline_reservoirs = [
    ("env_all", ["input_Tac", "input_u", "input_hs", "input_Rg"],),
    ("env_temp", ["input_Tac"]),
    ("env_humidity", ["input_hs"]),
    ("env_PAR", ["input_Rg"]),
    ("env_wind", ["input_u"]),
]


baseline_symbols = {
    "env_all": "All",
    "env_temp": "$T_{air}$",
    "env_humidity": "$RH$",
    "env_PAR": "$R_{g}$",
    "env_wind": "$u$",
}


heterogeneous_reservoirs = [
    ("state__het_all", (*state_variables,)),
]


state_names = {
    "state_An": "Net photosynthesis rate",
    "state_Tlc": "Leaf surface temperature",
    "state_Flux": "Water flow",
    "state_gb": "Boundary layer conductance",
    "state_gs": "Stomatal conductance",
    "state_psi_head": "Mean water potential",
    "state_E": "Transpiration rate",
    # Unknown
    # "state_FluxC": "",
    # "state_Eabs": "",
    # "state_Ei": "",
    # "state_u": "",
}

state_symbols = {
    "state_An": "$A_{n,leaf}$",
    "state_Tlc": "$T_{leaf}$",
    "state_Flux": "$F$",
    "state_gs": "$g_{s}$",
    "state_gb": "$g_{b}$",
    "state_psi_head": "$\Psi_{head}$",
    "state_E": "$E_{leaf}$",
    # Unknown
    # "state_FluxC": "",
    # "state_Eabs": "",
    # "state_Ei": "",
    # "state_u": "",
}

state_units = {
    "state_An": "µmol m$^{-2}$ s$^{-1}$",
    "state_Tlc": "°C",
    "state_Flux": "kg s$^{-1}$",
    "state_gs": "mol m$^{-2}$ s$^{-1}$",
    "state_gb": "mol m$^{-2}$ s$^{-1}$",
    "state_psi_head": "MPa",
    "state_E": "mol m$^{-2}$ s$^{-1}$",
    # Unknown
    # "state_FluxC": "",
    # "state_Eabs": "",
    # "state_Ei": "",
    # "state_u": "",
}


input_names = {
    "input_Tac": "Air temperature",
    "input_hs": "Air humidity",
    "input_Rg": "Solar shortwave irradiance",
    "input_u": "Wind speed",
}


input_symbols = {
    "input_Tac": "$T_{air}$",
    "input_hs": "$RH$",
    "input_Rg": "$R_{g}$",
    "input_u": "u",
}


input_units = {
    "input_Tac": "°C",
    "input_hs": "%",
    "input_Rg": "$W/m²$",
    "input_u": "m/s",
}

output_names = {
    "output_Rg": "Absorbed irradiance",
    "output_An": "Net carbon assimilation",
    "output_E": "Plant transpiration rate",
    "output_Tleaf": "Mean leaf temperature",
}


output_symbols = {
    "output_Rg": "$\Phi_{R_{g},plant}$",
    "output_An": "$A_{n,plant}$",
    "output_E": "$E_{plant}$",
    "output_Tleaf": "$T_{leaf,mean}$",
}


output_units = {
    "output_Rg": "W m$^{-2}$",
    "output_An": "µmol s$^{-1}$",
    "output_E": "g h$^{-1}$",
    "output_Tleaf": "°C",
}


best_reservoirs = [
    "state_Tlc",
    "state_psi_head",
    "state_E",
    # "state_Flux",
]
