#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Neural Estimation of Renyi Divergence (NERD), using gradient penalty.

(c) pantazis@iacm.forth.gr
    tsourtis@iacm.forth.gr
"""



import scipy.io
import tensorflow as tf  # tested with version 1.15.3
import numpy as np
import matplotlib.pyplot as plt
import os
from numpy import genfromtxt
from aux_functions2 import xavier_init

import csv
import sys

steps = int(sys.argv[1])  # number of steps
iteration = int(sys.argv[2])  # number of iterations
d = int(sys.argv[3])  # dimension
lam = float(sys.argv[4])  # lambda=beta+gamma
lam_gp = float(sys.argv[5])  # gradient penalty coefficient
mb_size = int(sys.argv[6])  # minibatch size
perc = sys.argv[7]  # subpopulation percentage (integer: 1-> 10%, 2-> 5%, 3-> 1%, 4->0.5%, 5->0.1%)
t = sys.argv[8]  # file id (integer)

# load data
#----------
# Healthy vs CBF
#fname = 'data/H3_H4_H5_H6_H7_vs_CBF_perc' + str(perc) + '_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16' # Healthy vs CBF
#fname = 'data/H3_H4_H5_H6_H7_HALF_SAMPLES_vs_CBF_perc' + str(perc) + '_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16' # Healthy vs CBF
# Healthy vs Healthy
#fname = 'data/H3_H4_H5_H6_H7_vs_Healthy_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16'  # Healthy vs Healthy
fname = 'data/H3_H4_H5_H6_H7_HALF_SAMPLES_vs_Healthy_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16'  # Healthy vs Healthy

data = scipy.io.loadmat(fname + '_iter_'+t+'.mat') 
#print('==============warning!! SWAPPED X<->Y (Y is CBF)==================')
x_ = np.array(data['Y']) 
y_ = np.array(data['X']) 


# choose for which alpha values Renyi divergence will be coputed
alpha = np.array([0.2, 0.5, 0.9]) #np.array(np.linspace(0.1, 1.1, 10))
No_alpha = alpha.shape[0]

# hyperparameters
if d==6:                      
    layers = [d, 16, 16, 8, 1]
elif d==15:                       
    layers = [d, 16, 16, 1] 
elif d==2:
    layers = [d, 16, 16, 8, 1]
elif d==10:
    layers = [d, 16, 16, 8, 1]
elif d == 16:
    layers = [d, 16, 16, 8, 1]
elif d==50:
    layers = [d, 64, 64, 32, 1]
elif d==1:                   
    layers = [d, 8, 8, 4, 1]
else:
    raise Exception("Check dimension and layers...")


# initialize
X = tf.placeholder(tf.float32, shape=[None, d])
Y = tf.placeholder(tf.float32, shape=[None, d])

def initialize_NN(layers):
    NN_W = []
    NN_b = []
    num_layers = len(layers)
    for l in range(0,num_layers-1):
        W = tf.Variable(xavier_init(size=[layers[l], layers[l+1]]))
        b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
        NN_W.append(W)
        NN_b.append(b)
    return NN_W, NN_b

D_W, D_b = initialize_NN(layers)
theta_D = [D_W, D_b] 

def discriminator(x):
    num_layers = len(D_W) + 1
    
    h = x
    for l in range(0,num_layers-2):
        W = D_W[l]
        b = D_b[l]
        h = tf.tanh(tf.add(tf.matmul(h, W), b))
    
    W = D_W[-1]
    b = D_b[-1]

    out =  tf.add(tf.matmul(h, W), b)   # unbounded!


    return out

D_real = discriminator(X)
D_fake = discriminator(Y)

config = tf.ConfigProto(device_count={'GPU': 0})
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

SF = 1000
D_loss_vals = np.zeros(shape=(No_alpha, iteration))


# estimate Renyi divergence
for j in range(No_alpha): # Each alpha is associated with another divergence
    beta = lam*(1-alpha[j]) 
    gamma = lam*alpha[j] 
    
    # variational representation:
    if beta == 0:
        D_loss_real = -tf.reduce_mean(D_real)
    else:
        max_val = tf.reduce_max((-beta) * D_real)
        D_loss_real = (1.0 / beta) * (tf.log(tf.reduce_mean(tf.exp((-beta) * D_real - max_val))) + max_val)

    if gamma == 0:
        D_loss_fake = tf.reduce_mean(D_fake)

    else:
        max_val = tf.reduce_max((gamma) * D_fake)
        D_loss_fake = (1.0 / gamma) * (tf.log(tf.reduce_mean(tf.exp(gamma * D_fake - max_val))) + max_val)

    D_loss = D_loss_real + D_loss_fake

    alpha_gp = tf.random_uniform(shape=[mb_size,1], minval=0., maxval=1.)
    interpolates = X + (alpha_gp*(Y - X)) X
    gradients = tf.gradients(discriminator(interpolates), [interpolates])[0] 
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1])) 
    gradient_penalty = tf.reduce_mean(tf.math.maximum(tf.zeros([slopes.shape[0]], dtype=tf.float32) ,(slopes-1.0))**2) #  (Lipschitz with k=1)

    total_loss = D_loss + lam_gp*gradient_penalty

    D_solver = tf.train.AdamOptimizer(learning_rate=0.01).minimize(total_loss, var_list=theta_D)
    #D_solver = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(total_loss, var_list=theta_D)


    for iter in range(iteration):  # iid runs (default: iteration == 1)
        print('Iteration: {}'.format(iter))
        sess.run(tf.global_variables_initializer())

        x = x_ 
        y = y_ 

        #initialize for plotting
        i = 0
        Pl_freq = 10
        D_loss_plot = np.zeros(shape=((np.rint(steps/Pl_freq)).astype(int), 1)) #because we writeout every Pl_freq

	# loop over training steps
        for it in range(steps):
            X_mb = x[np.random.randint(x.shape[0], size=mb_size), :]
            Y_mb = y[np.random.randint(y.shape[0], size=mb_size), :]

            _, D_loss_curr, D_tot_loss = sess.run([D_solver, D_loss, total_loss], feed_dict={X: X_mb, Y: Y_mb})

            if it % Pl_freq == 0:
                D_loss_curr = sess.run(D_loss, feed_dict={X: x, Y: y})
                D_loss_plot[i] = D_loss_curr
                i += 1

            if it % SF == 0:
                print('Iter: {}'.format(it))
                print('Renyi divergence: {}'.format(-lam*D_loss_curr))

                print()
                
        D_loss_curr = sess.run(D_loss, feed_dict={X: x, Y: y})
        D_loss_vals[j,iter] = -lam * D_loss_curr
   


    # -----------------------------------------------------------------------------
    # Plotting
    # -----------------------------------------------------------------------------
    if not os.path.exists('data/out_Lip_plots/'):
        os.makedirs('data/out_Lip_plots/')

    fig = plt.figure()
    #plt.plot(D_loss_plot)
    x_idx = np.linspace(0, steps, num=(np.rint(steps/Pl_freq)).astype(int))

    plt.plot(x_idx, D_loss_plot)
    plt.xlabel('Steps')
    plt.ylabel('D loss')
    plt.savefig('data/out_Lip_plots/cgan_Dloss' + str(j)+ 'perc_' + perc+ '_iter_' + t +'.png', bbox_inches='tight')
    plt.close(fig)



with open(fname + 'lambda_' + str(lam) + '_gp_' + str(lam_gp) + '_bs_' + str(mb_size) + '_nerd_iter'+t+'.csv', "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in D_loss_vals:
        writer.writerow(val)


print('Program terminated successfully')

