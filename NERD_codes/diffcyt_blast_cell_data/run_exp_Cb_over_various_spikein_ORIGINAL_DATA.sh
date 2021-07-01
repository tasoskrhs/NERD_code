#!/bin/sh

#  run_exp_batchsize.sh
#
# arguments: epochs,
#           iterations,
#           dimension,
#           batchsize,
#           percentage (int, 1-> 10%, 2-> 5%, 3-> 1%, 4->0.5%, 5->0.1%)
#           iter number (iters ressemble different random spike-in's of the same percentage)

#perc 20%
# 16 DIMS!!
python3 nerd_Cb_alphas_ORIGINAL_DATA.py 60000 1 16 4000 4 1 &
python3 nerd_Cb_alphas_ORIGINAL_DATA.py 60000 1 16 4000 4 2 &
python3 nerd_Cb_alphas_ORIGINAL_DATA.py 60000 1 16 4000 4 3 &
python3 nerd_Cb_alphas_ORIGINAL_DATA.py 60000 1 16 4000 4 4 &
python3 nerd_Cb_alphas_ORIGINAL_DATA.py 60000 1 16 4000 4 5

#python3 nerd_batchsize_16D_trainandpredict_alphas.py 40000 1 16 2000 1 6 &
#python3 nerd_batchsize_16D_trainandpredict_alphas.py 40000 1 16 2000 1 7 &
#python3 nerd_batchsize_16D_trainandpredict_alphas.py 40000 1 16 2000 1 8 &
#python3 qnerd_batchsize_16D_trainandpredict_alphas.py 40000 1 16 2000 1 9 &
#python3 nerd_batchsize_16D_trainandpredict_alphas.py 40000 1 16 2000 1 10

#perc 10%
#python3 nerd_batchsize_16D_trainandpredict.py 40000 1 16 2000 1 1 &
#python3 nerd_batchsize_16D_trainandpredict.py 40000 1 16 2000 1 2 &
#python3 nerd_batchsize_16D_trainandpredict.py 40000 1 16 2000 1 3 &
#python3 nerd_batchsize_16D_trainandpredict.py 40000 1 16 2000 1 4 &
#python3 nerd_batchsize_16D_trainandpredict.py 40000 1 16 2000 1 5 &
#python3 nerd_batchsize_16D_trainandpredict.py 40000 1 16 2000 1 6

#python3 nerd_batchsize_16D_trainandpredict.py 40000 1 16 2000 1 7 &
#python3 nerd_batchsize_16D_trainandpredict.py 40000 1 16 2000 1 8 &
#python3 nerd_batchsize_16D_trainandpredict.py 40000 1 16 2000 1 9 &
#python3 nerd_batchsize_16D_trainandpredict.py 40000 1 16 2000 1 10

#perc 5%
#python3 nerd_batchsize_16D_trainandpredict.py 40000 1 16 2000 2 1 &
#python3 nerd_batchsize_16D_trainandpredict.py 40000 1 16 2000 2 2 &
#python3 nerd_batchsize_16D_trainandpredict.py 40000 1 16 2000 2 3 &
#python3 nerd_batchsize_16D_trainandpredict.py 40000 1 16 2000 2 4 &
#python3 nerd_batchsize_16D_trainandpredict.py 40000 1 16 2000 2 5 &
#python3 nerd_batchsize_16D_trainandpredict.py 40000 1 16 2000 2 6

#python3 nerd_batchsize_16D_trainandpredict.py 40000 1 16 2000 2 7 &
#python3 nerd_batchsize_16D_trainandpredict.py 40000 1 16 2000 2 8 &
#python3 nerd_batchsize_16D_trainandpredict.py 40000 1 16 2000 2 9 &
#python3 nerd_batchsize_16D_trainandpredict.py 40000 1 16 2000 2 10

#perc 1%
#python3 nerd_batchsize_16D_trainandpredict.py 40000 1 16 2000 3 1 &
#python3 nerd_batchsize_16D_trainandpredict.py 40000 1 16 2000 3 2 &
#python3 nerd_batchsize_16D_trainandpredict.py 40000 1 16 2000 3 3 &
#python3 nerd_batchsize_16D_trainandpredict.py 40000 1 16 2000 3 4 &
#python3 nerd_batchsize_16D_trainandpredict.py 40000 1 16 2000 3 5 &
#python3 nerd_batchsize_16D_trainandpredict.py 40000 1 16 2000 3 6

#python3 nerd_batchsize_16D_trainandpredict.py 40000 1 16 2000 3 7 &
#python3 nerd_batchsize_16D_trainandpredict.py 40000 1 16 2000 3 8 &
#python3 nerd_batchsize_16D_trainandpredict.py 40000 1 16 2000 3 9 &
#python3 nerd_batchsize_16D_trainandpredict.py 40000 1 16 2000 3 10

#perc 0.5%
#python3 nerd_batchsize_16D_trainandpredict.py 40000 1 16 2000 4 1 &
#python3 nerd_batchsize_16D_trainandpredict.py 40000 1 16 2000 4 2 &
#python3 nerd_batchsize_16D_trainandpredict.py 40000 1 16 2000 4 3 &
#python3 nerd_batchsize_16D_trainandpredict.py 40000 1 16 2000 4 4 &
#python3 nerd_batchsize_16D_trainandpredict.py 40000 1 16 2000 4 5 &
#python3 nerd_batchsize_16D_trainandpredict.py 40000 1 16 2000 4 6

#python3 nerd_batchsize_16D_trainandpredict.py 40000 1 16 2000 4 7 &
#python3 nerd_batchsize_16D_trainandpredict.py 40000 1 16 2000 4 8 &
#python3 nerd_batchsize_16D_trainandpredict.py 40000 1 16 2000 4 9 &
#python3 nerd_batchsize_16D_trainandpredict.py 40000 1 16 2000 4 10

#perc 0.1%
#python3 nerd_batchsize_16D_trainandpredict.py 40000 1 16 2000 5 1 &
#python3 nerd_batchsize_16D_trainandpredict.py 40000 1 16 2000 5 2 &
#python3 nerd_batchsize_16D_trainandpredict.py 40000 1 16 2000 5 3 &
#python3 nerd_batchsize_16D_trainandpredict.py 40000 1 16 2000 5 4 &
#python3 nerd_batchsize_16D_trainandpredict.py 40000 1 16 2000 5 5 &
#python3 nerd_batchsize_16D_trainandpredict.py 40000 1 16 2000 5 6

#python3 nerd_batchsize_16D_trainandpredict.py 40000 1 16 2000 5 7 &
#python3 nerd_batchsize_16D_trainandpredict.py 40000 1 16 2000 5 8 &
#python3 nerd_batchsize_16D_trainandpredict.py 40000 1 16 2000 5 9 &
#python3 nerd_batchsize_16D_trainandpredict.py 40000 1 16 2000 5 10


wait  #needed for ending ALL concurrent runs before returning, so that the copy of .csv files is correct

