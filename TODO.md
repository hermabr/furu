# things i really want

- an overview of the full dag
     - the full dag given the existing experiments
     - the full dag given the code
- see all experiments and to filter
- rerun an experiment
- migration helper / show stale runs that are no longer valid
- see all dependencies and metadata about a particular experiment
    - option to rerun the full experiment either from the ui or by getting a code snippet that i can run
- create a new experiment with other hyperparameters or a sweep
- clean way to start a bunch of experiments
- discover all available runs in the code (is this even possible?) could maybe also be that i provide a json file with all my experiments so that i can make something similar to a wandb dashboard but where i also show how to achieve my results


## TODO

- [ ] add support for building to wheel
- [ ] remove all use of typing.Any
- [ ] remove the custom tailwind colors
- [ ] understand what absent status is
- [ ] make sure the end to end tests use actual data (e.g., running some script to generate data inside data-huldra)
- [ ] build wheels similar to python-react
- [ ] check if i always have access to the huldra.dashboard even if i only add only `uv add huldra` and not `uv add huldra[dashboard]`
- [ ] make the conftest tests use actual huldra code rather than manually making the json objects
  - [ ] make multiple different experiments with dependencies
- [ ] make the dashboard support going into an experiment where it should show the full config including the option to toggle in and out one part and click to go into that child experiment
- [ ] support seeing all children
- [ ] support showing which files and add viewer for simple formats like any parquet or json/jsonl
- [ ] see the logs of the experiments
- [ ] support rerunning an experiment
- [ ] support making graphs/charts. will need to decide if this needs to be from python or if we want it to be in react (might want python since this saves all the files)
- [ ] rename hexdigest and maybe make it a private method (e.g. \_hexdigest)
- [ ] show what is a subclass of what
- [ ] see nice dags of everything that depends on each other (maybe also with subclasses)
- [ ] add support for making experiments from the ui (and have it write the code to some folder in the directory before running)
- [ ] throw/assert/raise on unexpected behavior rather than trying to manually handle it
- [ ] support migrating, such as when adding a field
- [ ] show which experiments are version controlled and not
- [ ] move from hatchling to uv-build
- [ ] decide if dashboard-frontend should be inside src or not
