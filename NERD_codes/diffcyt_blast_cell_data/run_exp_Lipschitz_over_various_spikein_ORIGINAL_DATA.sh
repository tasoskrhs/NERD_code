#!/bin/sh

#  run exp for Lipschitz nerd
#  
# arguments: epochs,
#           iterations,
#           dimension,
#           lambda,
#           lambda_gp,
#           batchsize,
#           percentage (int, 1-> 10%, 2-> 5%, 3-> 1%, 4->0.5%, 5->0.1%)
#           iter number (iters ressemble different random spike-in's of the same percentage)


# H-H (half)
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 1 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 2 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 3 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 4 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 5

#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 6 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 7 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 8 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 9 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 10

#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 11 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 12 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 13 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 14 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 15

#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 16 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 17 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 18 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 19 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 20

#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 21 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 22 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 23 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 24 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 25

#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 26 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 27 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 28 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 29 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 30

#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 31 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 32 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 33 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 34 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 35

#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 36 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 37 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 38 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 39 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 40

#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 41 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 42 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 43 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 44 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 45

#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 46 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 47 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 48 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 49 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 50

#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 51 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 52 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 53 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 54 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 55

#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 56 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 57 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 58 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 59 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 60

#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 61 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 62 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 63 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 64 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 65

#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 66 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 67 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 68 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 69 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 70

#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 71 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 72 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 73 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 74 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 75

#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 76 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 77 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 78 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 79 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 80

#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 81 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 82 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 83 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 84 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 85

#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 86 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 87 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 88 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 89 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 90

#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 91 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 92 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 93 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 94 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 95

#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 96 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 97 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 98 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 99 &
#python3 nerd_Lipschitz_alphas_ORIGINAL_DATA.py 60000 1 16 1.0 0.1 4000 6 100


wait  #needed for ending ALL concurrent runs before returning, so that the copy of .csv files is correct
