#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
(c) pantazis@iacm.forth.gr
"""

import scipy.io
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from numpy import genfromtxt
from aux_functions2 import xavier_init

import csv
import sys

epochs = int(sys.argv[1])  # number of epochs
iteration = int(sys.argv[2])  # number of iterations
d = int(sys.argv[3])  # dimension
lam = float(sys.argv[4])  # lambda=beta+gamma
lam_gp = float(sys.argv[5])  # gradient penalty penalty hyperparameter
mb_size = int(sys.argv[6])  # read minibatch size
t = sys.argv[7]  # distance parameter t

# load data
fname = 'data/GMM_d_'+str(d)+'_'
#data = scipy.io.loadmat(fname + 'data.mat')
data = scipy.io.loadmat(fname + 'data_'+t+'.mat')
x_ = np.array(data['x'])
y_ = np.array(data['y'])

#params = scipy.io.loadmat(fname + 'params.mat')
params = scipy.io.loadmat(fname + 'params_'+t+'.mat')
alpha = np.array(params['alpha'])
No_alpha = alpha.shape[0]

# hyperparameters
#mb_size = 20000 #4000 #2000

if d==2:
    layers = [d, 16, 16, 8, 1]
elif d==10:
    layers = [d, 32, 32, 16, 1]
elif d==50:
    layers = [d, 64, 64, 32, 1]
elif d==1:                      #MINE
    layers = [d, 16, 16, 8, 1] #[d, 8, 8, 4, 1]  #[d, 4, 4, 2, 1] # 1, 8, 1

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
theta_D = [D_W, D_b] # python 2
#theta_D = D_W.copy() # python 3
#theta_D.extend(D_b)

def discriminator(x):
    num_layers = len(D_W) + 1
    
    h = x
    for l in range(0,num_layers-2):
        W = D_W[l]
        b = D_b[l]
        h = tf.tanh(tf.add(tf.matmul(h, W), b))
    
    W = D_W[-1]
    b = D_b[-1]
    #out = 50.0 * tf.nn.tanh(tf.add(tf.matmul(h, W), b) / 50.0) # coeff ->  Bounded (by coeff) and cont (C_b function space)
    out =  tf.add(tf.matmul(h, W), b)   # unbounded!
    #out = tf.nn.tanh(tf.add(tf.matmul(h, W), b))

    return out

D_real = discriminator(X)
D_fake = discriminator(Y)

config = tf.ConfigProto(device_count={'GPU': 0})
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

SF = 1000
D_loss_vals = np.zeros(shape=(No_alpha, iteration))

eval_pts = np.expand_dims(np.linspace(-10, 10, 201), 1)  # support of x
discr_vals = np.zeros(shape=(No_alpha, eval_pts.shape[0]))

# estimate Renyi divergence
for j in range(No_alpha):
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
    interpolates = X + (alpha_gp*(Y - X)) #mine. in 1-D this is the analog to x+ DeltaX
    gradients = tf.gradients(discriminator(interpolates), [interpolates])[0]  # in 1-D f(x+DeltaX) -f(x)/ (DeltaX)
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1])) # ||f(x+DeltaX) -f(x)|| /DeltaX
    #print('slopes type', type(slopes))  #class 'tensorflow.python.framework.ops.Tensor'
    #print('slopes size', slopes.shape[0] ) # 2000
    print('-------------')
    #gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)  # ||f(x+DeltaX) -f(x)|| /DeltaX  < 1  (Lipschitz with k=1)
    #gradient_penalty = tf.reduce_mean(tf.math.maximum(tf.zeros([slopes.shape[0]], dtype=tf.float32) ,(slopes-1.))**2)  #||f(x+DeltaX) -f(x)|| /DeltaX  < 1  (Lipschitz with k=5)
    #gradient_penalty = tf.reduce_mean(tf.math.maximum(np.zeros(shape=(slopes.shape[0])),(slopes-5.))**2)  # doesn't work!
    #gradient_penalty = tf.reduce_mean(tf.math.maximum(tf.zeros([slopes.shape[0]], dtype=tf.float32) ,(slopes-5.))**2)  #||f(x+DeltaX) -f(x)|| /DeltaX  < 1  (Lipschitz with k=5)
    gradient_penalty = tf.reduce_mean(tf.math.maximum(tf.zeros([slopes.shape[0]], dtype=tf.float32) ,(slopes-1.))**2)  #||f(x+DeltaX) -f(x)|| /DeltaX  < 1  (Lipschitz with k=1)

    total_loss = D_loss + lam_gp*gradient_penalty

    # D_solver = tf.train.AdamOptimizer().minimize(total_loss, var_list=theta_D)  # learning_rate = 0.001
    D_solver = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(total_loss, var_list=theta_D)

    for iter in range(iteration):
        print('Iteration: {}'.format(iter))
        sess.run(tf.global_variables_initializer())

        x = x_[np.random.randint(x_.shape[0], size=int(0.8*x_.shape[0])), :]
        y = y_[np.random.randint(y_.shape[0], size=int(0.8*x_.shape[0])), :]
        #initialize for plotting
        i = 0
        Pl_freq = 10
        D_loss_plot = np.zeros(shape=((np.rint(epochs/Pl_freq)).astype(int), 1)) #because we writeout every Pl_freq

        for it in range(epochs):
            X_mb = x[np.random.randint(x.shape[0], size=mb_size), :]
            Y_mb = y[np.random.randint(y.shape[0], size=mb_size), :]

            _, D_loss_curr, D_tot_loss = sess.run([D_solver, D_loss, total_loss], feed_dict={X: X_mb, Y: Y_mb})

            if it % Pl_freq == 0:
                D_loss_plot[i] = D_loss_curr
                i += 1

            if it % SF == 0:
                print('Iter: {}'.format(it))
                print('Renyi divergence: {}'.format(-lam*D_loss_curr))

                print()
                
        D_loss_curr = sess.run(D_loss, feed_dict={X: x, Y: y})
        D_loss_vals[j,iter] = -lam * D_loss_curr
   
    #save the discriminator of the LAST iteration
    predictions_real, predictions_fake = sess.run([D_real, D_fake], feed_dict={X: eval_pts, Y: X_mb})
    discr_vals[j, :] = predictions_real.flatten()


    # -----------------------------------------------------------------------------
    # Plotting
    # -----------------------------------------------------------------------------
    if not os.path.exists('data/out_GMM_Lip_plots/'):
        os.makedirs('data/out_GMM_Lip_plots/')

    fig = plt.figure()
    #plt.plot(D_loss_plot)
    x_idx = np.linspace(0, epochs, num=(np.rint(epochs/Pl_freq)).astype(int))

    plt.plot(x_idx, D_loss_plot)
    plt.xlabel('Epochs')
    plt.ylabel('D loss')
    plt.savefig('data/out_GMM_Lip_plots/cgan_Dloss' + str(j)+ 'w_' + t +'.png', bbox_inches='tight')
    plt.close(fig)


#with open(fname+'lambda_'+str(lam)+'_gp_'+str(lam_gp)+'_bs_'+str(mb_size)+'_nerd.csv', "w") as output:
with open(fname + 'lambda_' + str(lam) + '_gp_' + str(lam_gp) + '_bs_' + str(mb_size) + '_nerd_'+t+'.csv', "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in D_loss_vals:
        writer.writerow(val)

with open(fname + 'lambda_' + str(lam) + '_gp_' + str(lam_gp) + '_bs_' + str(mb_size) + '_nerd_'+t + 'discrim' + '.csv', "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in discr_vals:
        writer.writerow(val)

