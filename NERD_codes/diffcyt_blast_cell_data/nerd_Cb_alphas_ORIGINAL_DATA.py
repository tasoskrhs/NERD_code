#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
(c) pantazis@iacm.forth.gr
"""
# Version based on "nerd_batchsize_16D_trainandpredict_alphas.py" with: different input file names, layers.

import scipy.io
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from numpy import genfromtxt
from aux_functions2 import xavier_init
from aux_functions2 import plot #MINE

import csv
import sys

epochs = int(sys.argv[1])  # number of epochs
iteration = int(sys.argv[2])  # number of iterations
d = int(sys.argv[3])  # dimension
mb_size = int(sys.argv[4])  # batch size
perc = sys.argv[5]
t = sys.argv[6]
print(epochs)
print(iteration)
print(d)
print(mb_size)
print(perc)
print(t)

# load data
fname = 'data/H3_H4_H5_H6_H7_vs_CBF_perc' + str(perc) + '_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16'
fname_sav = 'data/' #mine
data = scipy.io.loadmat(fname + '_iter_'+t+'.mat') #scipy.io.loadmat(fname + 'data_'+t+'.mat')
#data = scipy.io.loadmat(fname + 'data_0.4.mat')
print('==============warning!! SWAPPED X<->Y (Y is CBF)==================')
x_ = np.array(data['Y']) #(data['X'])#
y_ = np.array(data['X']) #(data['Y'])#

#params = scipy.io.loadmat(fname + 'params_0.4.mat')
#params = scipy.io.loadmat(fname + 'params_'+t+'.mat')
#alpha = np.array(params['alpha'])
#No_alpha = alpha.shape[0]

#alpha =  0.5 #0.99 # 0.01 # 0.99 # #hardcoded, 0.99 <-> DKL, 0.5 <-> Hellinger
#No_alpha = 1
#alpha = np.array([0.0, 1.0])
#alpha = np.array(np.linspace(0.1, 1.1, 10))#np.array(np.linspace(0.1, 2, 20))#((np.linspace(-2, 2, 42)))#
print('a= 0.1, 0.2,... 0.9!! ')
alpha = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
No_alpha = alpha.shape[0]
print(alpha[1])
print(No_alpha)


# hyperparameters
if d==6:                       # <-- diffcyt real data
    layers = [d, 16, 16, 1] #[d, 128, 1] #[d,16,16,8,1]#[d, 32, 16, 1]#[d, 16, 1] #[d, 16, 8, 1] #[d, 16, 16, 8, 1] #[d, 8, 8, 4, 1]
elif d==15:                       # <-- diffcyt real data
    layers = [d, 16, 16, 1] #[d, 128, 1] #[d,16,16,8,1]#[d, 32, 16, 1]#[d, 16, 1] #[d, 16, 8, 1] #[d, 16, 16, 8, 1] #[d, 8, 8, 4, 1]
elif d==2:
    layers = [d, 16, 16, 8, 1]
elif d==10:
    layers = [d, 16, 16, 8, 1]#[d, 32, 32, 16, 1]
elif d == 16:
    layers = [d, 16, 16, 8, 1]
elif d==50:
    layers = [d, 64, 64, 32, 1]
elif d==1:                      #MINE
    layers = [d, 8, 8, 4, 1]
else:
    raise Exception("Check dimension and layers...")

lam = 1.0 # lambda=beta+gamma

# initialize
X = tf.placeholder(tf.float32, shape=[None, d])
Y = tf.placeholder(tf.float32, shape=[None, d])

def initialize_NN(layers):
    NN_W = []
    NN_b = []
    num_layers = len(layers)
    for l in range(0,num_layers-1):
        #W = tf.Variable(xavier_init(size=[layers[l], layers[l+1]])) # <------------------------
        W = tf.Variable(xavier_init(size=[layers[l], layers[l+1]]), name="W")
        b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
        NN_W.append(W)
        NN_b.append(b)
    return NN_W, NN_b

D_W, D_b = initialize_NN(layers)
#print(type(D_W))

theta_D = [D_W, D_b] # python 2
#theta_D = D_W.copy() # python 3
#theta_D.extend(D_b)

def discriminator(x):
    num_layers = len(D_W) + 1
    
    h = x  #h = [x, x**2] and fix D_w dimension
    for l in range(0,num_layers-2):
        W = D_W[l]
        b = D_b[l]
        h = tf.tanh(tf.add(tf.matmul(h, W), b))
    
    W = D_W[-1]
    b = D_b[-1]
    #out = 1.0 * tf.nn.tanh(tf.add(tf.matmul(h, W), b) / 1.0)
    out = 5.0 * tf.nn.tanh(tf.add(tf.matmul(h, W), b) / 5.0)
    #out = 50.0 * tf.nn.tanh(tf.add(tf.matmul(h, W), b) / 50.0)

    return out

D_real = discriminator(X)
D_fake = discriminator(Y)


config = tf.ConfigProto(device_count={'GPU': 0})
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

SF = 1000
D_loss_vals =  np.zeros(shape=(No_alpha, iteration))  # np.zeros(shape=(1, iteration)) <----------
#alpha_05 = 0.5 <----------

#eval_pts = np.expand_dims(np.linspace(-10, 10, 201), 1)  # support of x
#discr_vals = np.zeros(shape=(No_alpha, eval_pts.shape[0]))


# estimate Renyi divergence
for j in range(No_alpha):  # range(1): # <-------------------
    #print('j: {}'.format(j))
    #print('alpha[j]: {}'.format(alpha[j]))
    beta = lam*(1-alpha[j]) #lam*(1-alpha) #  # lam*(1-alpha_05)  <-----------
    gamma = lam*alpha[j] # lam*alpha # lam*alpha_05  #  <----------
    
    # variational representation:
    if beta == 0:
        D_loss_real = tf.reduce_mean(D_real)
        print('entree beta is zero' )
    else:
        max_val = tf.reduce_max((-beta) * D_real)
        D_loss_real = (1.0 / beta) * (tf.log(tf.reduce_mean(tf.exp((-beta) * D_real - max_val))) + max_val)

    if gamma == 0:
        D_loss_fake = tf.reduce_mean(D_fake)
        print('entree gamma is zero')

    else:
        max_val = tf.reduce_max((gamma) * D_fake)
        D_loss_fake = (1.0 / gamma) * (tf.log(tf.reduce_mean(tf.exp(gamma * D_fake - max_val))) + max_val)

    D_loss = D_loss_real + D_loss_fake
    total_loss = D_loss

    #D_solver = tf.train.AdamOptimizer().minimize(total_loss, var_list=theta_D)
    #D_solver = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(total_loss, var_list=theta_D)
    D_solver = tf.train.AdamOptimizer(learning_rate=0.01).minimize(total_loss, var_list=theta_D)

    for iter in range(iteration):
        print('Iteration: {}'.format(iter))
        sess.run(tf.global_variables_initializer())

        x = x_ #x_[np.random.randint(x_.shape[0], size=int(0.8*x_.shape[0])), :]
        y = y_ #y_[np.random.randint(y_.shape[0], size=int(0.8*x_.shape[0])), :]

        # initialize for plotting
        i = 0
        Pl_freq = 10
        D_loss_plot = np.zeros(shape=((np.rint(epochs / Pl_freq)).astype(int), 1))  # because we writeout every Pl_freq

        for it in range(epochs):
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

    #save the discriminator of the LAST iteration
    #predictions_real, predictions_fake = sess.run([D_real, D_fake], feed_dict={X: eval_pts, Y: X_mb})

    #discr_vals[j,:] = predictions_real.flatten() #predictions_real[:,1] # predictions_real[:,1]   #predictions_real.reshape((3,1))

    # -----------------------------------------------------------------------------
    # Plotting
    # -----------------------------------------------------------------------------
    if not os.path.exists('data/out_BS_plots/'):
        os.makedirs('data/out_BS_plots/')

    fig = plt.figure()
    #plt.plot(D_loss_plot)
    x_idx = np.linspace(0, epochs, num=(np.rint(epochs / Pl_freq)).astype(int))
    plt.plot(x_idx, D_loss_plot)
    plt.xlabel('Epochs')
    plt.ylabel('D loss')
    plt.savefig('data/out_BS_plots/cgan_Dloss' + str(j) + 'perc_' + perc+ '_iter_' + t +'.png', bbox_inches='tight')
    plt.close(fig)
   

with open(fname+'_lambda_'+str(lam)+'_bs_'+str(mb_size)+'_nerd_iter'+t+'.csv', "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in D_loss_vals:
        writer.writerow(val)

#with open(fname + 'lambda_' + str(lam) + '_bs_' + str(mb_size) + '_nerd_' + t + 'discrim' + '.csv', "w") as output:
#    writer = csv.writer(output, lineterminator='\n')
#    for val in discr_vals:
#        writer.writerow(val)


print('program terminated succesfully')