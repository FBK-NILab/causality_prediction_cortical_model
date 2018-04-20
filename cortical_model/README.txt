FILES
=====

Each folder [simulation_name] has a [simulation_name].c, [simulation_name].m, OU_euler_seed.m and an output folder.

[simulation_name]   - folder containing one simulation according to the given causality scenario, e.g. "model_3_areas_1to2_3to2_1to3" means "1->2, 3->2, 1->3".
[simulation_name].c - C code of the given simulation
[simulation_name].m - MATLAB code that sets all the simulation parameters, makes it run in parallel and saves the output
OU_euler_seed.m     - code that simulates Ornstein–Uhlenbeck process
/output             - folder with the file that contains the results of the simulation

OUTPUT DATA STRUCTURE
=====================
in the output .mat file there is a matrix called output with the following properties:
output = matrix[#samples X #networks X time] where:
        #samples  - the number of simulation runs performed
        #networks - number of networks - in our case always 3
        time      - time of the time series

        Example: output(1, 2, 3) gives the time series values of the simulation run #1 of the network 2 at 3ms on the time axis.


INSTRUCTIONS
============

In order to run all the simulations it is necessary to complete the following steps:

1) Compile all the *.c files with MEX (from the MATLAB environment)
2) Set the variable "numWorkers" in [simulation_name].m to the intended number of processes that should perform the parallel computation (the MATLAB parpool needs to be properly configured for large number of cores)
    -rule of thumb: if you have a CPU with <= 8 cores, set numWorkers = #cores; otherwise set numWorkers = #cores - 1
3) Run the [simulation_name].m
