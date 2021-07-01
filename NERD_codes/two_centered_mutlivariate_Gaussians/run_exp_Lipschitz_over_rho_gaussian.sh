#!/bin/sh

#  run exp for Lipschitz nerd
#  
# use different rho values for smaller mode in gaussian distibution p
#

#python3 nerd_multivar_Lipschitz_trainandpredict.py 20000 10 20 1.0 0.1 4000  -0.9 &
#python3 nerd_multivar_Lipschitz_trainandpredict.py 20000 10 20 1.0 0.1 4000 -0.7 &
#python3 nerd_multivar_Lipschitz_trainandpredict.py 20000 10 20 1.0 0.1 4000 -0.5 &
#python3 nerd_multivar_Lipschitz_trainandpredict.py 20000 10 20 1.0 0.1 4000 -0.3 &
#python3 nerd_multivar_Lipschitz_trainandpredict.py 20000 10 20 1.0 0.1 4000 -0.1

python3 nerd_multivar_Lipschitz_trainandpredict.py 40000 10 50 1.0 0.1 40000  0.1 &
python3 nerd_multivar_Lipschitz_trainandpredict.py 40000 10 50 1.0 0.1 40000 0.3 &
python3 nerd_multivar_Lipschitz_trainandpredict.py 40000 10 50 1.0 0.1 40000 0.5 &
python3 nerd_multivar_Lipschitz_trainandpredict.py 40000 10 50 1.0 0.1 40000 0.7 &
python3 nerd_multivar_Lipschitz_trainandpredict.py 40000 10 50 1.0 0.1 40000 0.9


wait  #needed for ending ALL concurrent runs before returning, so that the copy of .csv files is correct
