"""
This file contains constants defining which inputs, outputs and state variables 
of the WheatFspm dataset are to be used for reservoir computing experiments.
"""

hydroshoot_inputs = (
    "input_air_temperature",
    "input_humidity",
    # "input_Wind",  # bad target because it only provides a new record every 24h
    "input_PARi",
)

hydroshoot_outputs = (
    "output__axes__Total_Transpiration",
    "output__axes__C_exudated",
    "output__axes__SAM_temperature",
)

hydroshoot_state = (
    "state__An",
    "state__Transpiration",
    "state__Ts",
    "state__gs",
    # "state__S_Sucrose",
)

exclusive_outputs = (
    "output__axes__delta_teq",
    "output__axes__sum_respi_shoot",
    "output__organ_roots__N_exudation",
)

exclusive_state = (
    "state__Ag",
    "state__Tr",
    "state__Rd",
    "state__sum_respi",
    "state__PARa",
    # "state__sucrose",
    # "state__Photosynthesis",
)


input_targets = [*hydroshoot_inputs]
output_targets = [*hydroshoot_outputs, *exclusive_outputs]


targets = [*input_targets, *output_targets]
state_variables = [*hydroshoot_state, *exclusive_state]


max_time_step = {
    "NEMA_H0": 696,  # originally 696
    "NEMA_H3": 864 - 24,  # originally 864
    "NEMA_H15": 912 - 24,  # originally 912
}


baseline_reservoirs = [
    ("env_all", ["input_air_temperature", "input_humidity", "input_PARi"],),
    ("env_temp", ["input_air_temperature"]),
    ("env_humidity", ["input_humidity"]),
    ("env_PAR", ["input_PARi"]),
]


baseline_symbols = {
    "env_all": "Env",
    "env_temp": "$T_{air}$",
    "env_humidity": "$RH$",
    "env_PAR": "$I_{PAR}$",
}


heterogeneous_reservoirs = [
    ("state__het_all", (*state_variables,)),
]

output_names = {
    "output__axes__Total_Transpiration": "Total transpiration",
    "output__axes__C_exudated": "Exudated C (axes)",
    "output__axes__SAM_temperature": "SAM temperature (axes)",
    "output__axes__delta_teq": "temperature-compensated time (axes)",
    "output__axes__sum_respi_shoot": "Total respiration (shoots)",
    "output__organ_roots__N_exudation": "Exudated N (roots)",
}

output_symbols = {
    "output__axes__Total_Transpiration": "",
    "output__axes__C_exudated": "",
    "output__axes__SAM_temperature": "",
    "output__axes__delta_teq": "",
    "output__axes__sum_respi_shoot": "",
    "output__organ_roots__N_exudation": "",
}

input_names = {
    "input_air_temperature": "Air temperature",
    "input_humidity": "Air humidity",
    "input_PARi": "Photosynthetically active radiation",
}

input_symbols = {
    "input_air_temperature": "$T_{air}$",
    "input_humidity": "$RH$",
    "input_PARi": "$I_{{PAR}}$",
}


state_names = {
    "state__Ts": "Organ surface temperature",
    "state__Tr": "Organ surface transpiration rate",
    "state__Transpiration": "Organ transpiration rate",
    "state__Ag": "Gross photosynthesis rate",
    "state__An": "Net photosynthesis rate",
    "state__gs": "Stomatal conductance",
    "state__Rd": "Mitochondrial respiration rate of organ in light",
    "state__PARa": "Absorbed PAR",
    "state__sum_respi": "Sum of respiration (element)",
    # Unknown
    # "state__S_Sucrose": "state__S_Sucrose",
    # "state__sucrose": "state__sucrose",
    # "state__Photosynthesis": "state__Photosynthesis",
}


state_symbols = {
    "state__Ts": "$T_{s}$",
    "state__Tr": "$Tr_{tp,i}$",
    "state__Transpiration": "Organ transpiration rate",
    "state__Ag": "$Ag_{tp,i}$",
    "state__An": "$An_{tp,i}$",
    "state__gs": "$g_{s}$",
    "state__Rd": "$Rd$",
    "state__PARa": "$PAR_{a}$",
    "state__sum_respi": "Sum of respiration (element)",
    # Unknown
    # "state__S_Sucrose": "state__S_Sucrose",
    # "state__sucrose": "state__sucrose",
    # "state__Photosynthesis": "state__Photosynthesis",
}


state_units = {
    "state__Ts": "°C",
    "state__Tr": "mmol H$_{2}$0 m$^{-2}$ s$^{-1}$",
    "state__Transpiration": "mmol H$_{2}$0 s$^{-1}$",
    "state__Ag": "µmol m$^{-2}$ s$^{-1}$",
    "state__An": "µmol m$^{-2}$ s$^{-1}$",
    "state__gs": "mol m$^{-2}$ s$^{-1}$",
    "state__Rd": "µmol C h$^{-1}$",
    "state__PARa": "µmol m$^{-2}$ s$^{-1}$",
    "state__sum_respi": "µmol C",
    # Unknown
    # "state__S_Sucrose": "",
    # "state__sucrose": "",
    # "state__Photosynthesis": "",
}

measurable_reservoirs = [
    "state__An",
    "state__Ts",
    "state__Tr",
    "state__Transpiration",
    "state__Rd",
    "state__gs",
    "state__sum_respi",
]


best_reservoirs = [
    "state__Ts",
    "state__Tr",
    "state__Rd",
]

