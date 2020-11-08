'''
training learning rate visualization tools for caffe and etc.
by sangkny
'''
import numpy as np
import matplotlib.pyplot as plt


def get_triangular_lr(iteration, stepsize, base_lr, max_lr):
    """Given the inputs, calculates the lr that should be applicable for this iteration"""
    cycle = np.floor(1 + iteration/(2  * stepsize))
    x = np.abs(iteration/stepsize - 2 * cycle + 1)
    lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1-x))
    return lr

def get_caffe_lr(mode, iteration, stepsize, base_lr, max_lr, gamma, power, max_iter =100000):
    """Given the inputs, calculates the lr that should be applicable for this iteration"""

    if(mode =='step'):
        lr = base_lr*pow(gamma, (np.floor(iteration/stepsize)))
    elif (mode == 'inv'):
        lr = base_lr * pow((1 + gamma * iteration), (-power))
    elif(mode == 'sigmoid'):
        lr = base_lr * ( 1/(1 + np.exp(-gamma * (iteration - stepsize))))
    elif(mode == 'exp'):
        base_lr * pow(gamma, iteration)
    elif(mode == 'fixed'):
        lr = base_lr
    elif(mode=='poly'):
        lr = base_lr * pow((1 - iteration/max_iter),  (power))
    elif(mode == 'multistep'):
        lr = 0
        print('it is not implemented yet')
    else:
        print('please check the mode.')

    return lr

# Demo of how the LR varies with iterations
num_iterations = 200000
max_iterations = num_iterations
#------------ inverse
base_lr = 0.001 # was 0.001 for parking
max_lr = 0.01 # which is not used
gamma = 0.0001 # was 0.0001
momentum = 0.9
power = 0.1
#---------------------
# # step--------------
# base_lr = 0.001
# max_lr = 0.001
# gamma = 0.0001
# monentum = 0.9
# power = 0.75
# -------------------
stepsize = 20000

lr_trend = list()
lr_caffe = list()
mode = 'inv'
print('CAFFE LR mode:' + mode)
for iter in range(num_iterations):
    lr = get_triangular_lr(iter, stepsize, base_lr, max_lr)
    # Update your optimizer to use this learning rate in this iteration
    lr_trend.append(lr)
    lr1 = get_caffe_lr(mode=mode, iteration=iter,stepsize=stepsize, base_lr=base_lr, max_lr=max_lr, gamma=gamma, power=power, max_iter=max_iterations)
    lr_caffe.append(lr1)

lrInfo = 'mode:{},base_lr:{},max_lr:{},gamma:{},power:{}'.format(mode, base_lr,max_lr, gamma, power)
#plt.plot(lr_trend)
plt.plot(lr_caffe)
plt.title(lrInfo)
plt.show()