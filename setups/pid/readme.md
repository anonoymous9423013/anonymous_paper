# PID Test Setup

<img src="../figs/pid.png" width="100%">

This test setup was developed from scratch by the authors of the paper.

To generate new test scenarios, simply run the `mh_test_generator.py` script. This script will generate 1000 scenarios and save them in the `mh_flaky_test_1.txt` file.

To run the test scenarios, simply run the `mh_tester.py` script. This script will run the 1000 scenarios (each scenario 10 times) and save the results in the `output` folder.
