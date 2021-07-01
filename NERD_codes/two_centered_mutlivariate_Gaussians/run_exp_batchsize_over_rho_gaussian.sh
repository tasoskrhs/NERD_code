#!/bin/sh

#  run_exp_batchsize.sh
#  
# use different rho values for smaller mode in gaussian distibution p

# d=4
#python3 nerd_multivar_gaussian_batchsize_trainandpredict.py 20000 10 20 4000 -0.9 &
#python3 nerd_multivar_gaussian_batchsize_trainandpredict.py 20000 10 20 4000 -0.7 &
#python3 nerd_multivar_gaussian_batchsize_trainandpredict.py 20000 10 20 4000 -0.5 &
#python3 nerd_multivar_gaussian_batchsize_trainandpredict.py 20000 10 20 4000 -0.3 &
#python3 nerd_multivar_gaussian_batchsize_trainandpredict.py 20000 10 20 4000 -0.1

python3 nerd_multivar_gaussian_batchsize_trainandpredict.py 40000 5 50 40000 0.1 &
python3 nerd_multivar_gaussian_batchsize_trainandpredict.py 40000 5 50 40000 0.3 &
python3 nerd_multivar_gaussian_batchsize_trainandpredict.py 40000 5 50 40000 0.5 &
python3 nerd_multivar_gaussian_batchsize_trainandpredict.py 40000 5 50 40000 0.7 &
python3 nerd_multivar_gaussian_batchsize_trainandpredict.py 40000 5 50 40000 0.9

wait  #needed for ending ALL concurrent runs before returning, so that the copy of .csv files is correct
