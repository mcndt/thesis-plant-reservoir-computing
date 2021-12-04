# thesis-cyborg-plants

## HydroShoot

I added a forked HydroShoot as a submodule to this project. All experiment templates live in this separate repository.

## Running batch experiments

You can use the following script to run batches of experiments using a multiprocessing worker pool:

```bash
# run from repository root
python -m src.run.from_json <json_file>
```

Example of a JSON file describing an experiment:

```json
{
    "name": "gdc_can1_6days", // name of the output directory
    "path": "hydroshoot-prc-experiments/prc_experiment_templates/gdc_can1_nodeficit", // path to the hydroshoot experiment directory. Must contain a python script containing the simulation called "sim.py"
    "sdate": "2012-08-01", // first day of the range
    "edate": "2012-08-03", // last day of the range
    "length": "6",
    "extend": true // if false, the last experiment will end on 'edate'. If true, the last experiment starts on 'edate'.
}
```