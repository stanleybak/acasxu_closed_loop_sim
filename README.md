# ACASXu Closed Loop Simulation Falsification Benchmark
Closed-loop simulation code using ACAS Xu neural networks for collision avoidance.

This repo contains different versions of in-plane flight with ACAS Xu for collision avoidance. Generally, the networks are activated by the ownship every 2 seconds to choose a command. The intruder also can adjust its command during flight.

## Features:

* Nice visualization capability (and mp4 export).
* Initial state, chosen randomly, can also be rejected if ownship neural network command is not clear-of-conflict.
* Simulations stop once the distance between airfraft is increasing. This means that simulations may take different amounts of time to run.
* Includes random initial state as well as random intruder commands. If only straight-line intruder commands are desired, intruder_cmd_list can be set to all 0's (clear-of-conflict, fly straight).
* Goal: minimize distance between the two aircraft. Under 500 ft would be a violation. If intruder is faster the ownship, the property should always be possible to violate (assuming intruder can maneuver). Interuder velocity can therefore be used to tune the difficulty of the benchmark.

## Usage:

Both `acasxu_dubins.py`  and `parallel_acasxu_dubins.py` have the same flags as below:

    usage: acasxu_dubins.py [-h] [--save-mp4]

    Run ACASXU Dublins model simulator.

    optional arguments:
        -h, --help  show this help message and exit
        --save-mp4  Save plotted mp4 files to disk.



TOOD list:
* add different plant models
* randomize ownship and intruder velocities within valid neural network range (parameter uncertainty)
* add command-line arguments for different versions of the benchmark
* add replay capability and save mp4 from command line
* add command-line usage to the readme
* find head-on example with same velocity and mirroring commands for paper.
* add different "desired" ownship commands, rather than straight
* add intruder (and maybe ownship) commands to be anything between -3 and 3 degrees per second, rather than just in `[-3, -1.5, 0, 1.5, 3.0]`
* add input quantization version (see https://arxiv.org/abs/2108.07961 for example values)
