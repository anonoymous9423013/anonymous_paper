# Comp Test Setup

<img src="../figs/comp.png" width="100%">

This test setup is based on the implementation of the [Cyber-Physical Systems Testing Tool Competition](https://github.com/sbft-cps-tool-competition/cps-tool-competition).

- `scenarios.csv` contains 1000 generated scenrios. This file should be copied in the root folder of the competition repository. To generate more scenarios, you can use the `mh_scenario_generator.py` script.

- Our road generator is implemented in `mh_runner.py`. This script should be copied in the `sample_test_generators` folder of the competition repository.

- To start simulations, simply enter the following command in the competition repository root folder:

```bash
python competition.py --time-budget 3600 --executor beamng --map-size 300 --module-name sample_test_generators.mh_runner --beamng-user C:\Research\User --beamng-home C:\Research\BeamNG.tech.v0.26.2.0\BeamNG.tech.v0.26.2.0  --class-name MHTestGenerator --log-to C:\Research\Logs\logs.txt
```

Don't forget to change the paths of beamng home and user to your own paths.
