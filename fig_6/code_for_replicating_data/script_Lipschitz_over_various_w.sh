#!/bin/sh

#  Lipschitz neRd computations for the real data sets
#  
# arguments: training steps,
#           iterations,
#           dimension,
#           lambda,
#           lambda_gp,
#           batchsize,
#           percentage (int, 1-> 10%, 2-> 5%, 3-> 1%, 4->0.5%, 5->0.1%)
#           iter number (iters ressemble different random spike-in's of the same percentage)


# Each line is an iid run. Load the input files (lines 35-40) according to the chosen experiment
python3 nerd_Lipschitz_alphas_real_data.py 60000 1 16 1.0 0.1 4000 6 1 &
python3 nerd_Lipschitz_alphas_real_data.py 60000 1 16 1.0 0.1 4000 6 2 &
python3 nerd_Lipschitz_alphas_real_data.py 60000 1 16 1.0 0.1 4000 6 3 &
python3 nerd_Lipschitz_alphas_real_data.py 60000 1 16 1.0 0.1 4000 6 4 &
python3 nerd_Lipschitz_alphas_real_data.py 60000 1 16 1.0 0.1 4000 6 5

python3 nerd_Lipschitz_alphas_real_data.py 60000 1 16 1.0 0.1 4000 6 6 &
python3 nerd_Lipschitz_alphas_real_data.py 60000 1 16 1.0 0.1 4000 6 7 &
python3 nerd_Lipschitz_alphas_real_data.py 60000 1 16 1.0 0.1 4000 6 8 &
python3 nerd_Lipschitz_alphas_real_data.py 60000 1 16 1.0 0.1 4000 6 9 &
python3 nerd_Lipschitz_alphas_real_data.py 60000 1 16 1.0 0.1 4000 6 10

# For additional runs, either add a for loop or copy the lines above with different "iter number"


wait  #needed for ending ALL concurrent runs before returning, so that the copy of .csv files is correct
