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
    "state__S_Sucrose",
    "state__Ts",
    "state__gs",
)

exclusive_outputs = (
    "output__axes__delta_teq",
    "output__axes__sum_respi_shoot",
    "output__organ_roots__N_exudation",
)

exclusive_state = (
    "state__Ag",
    "state__Tr",
    "state__sucrose",
    "state__Rd",
    "state__sum_respi",
    "state__Photosynthesis",
    "state__PARa",
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

