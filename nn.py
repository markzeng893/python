import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

# Q 2.1.2
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    #W, b = None, None
    W = np.random.uniform(-1.0, 1.0, (in_size, out_size)) * ( np.sqrt(6) / np.sqrt(in_size + out_size))
    b = np.zeros(out_size)
    
    params['W' + name] = W
    params['b' + name] = b

# Q 2.2.1
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    res = 1.0 / (1.0 + np.exp(-x))
    return res

# Q 2.2.1
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    #pre_act, post_act = None, None
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]

    pre_act = (X @ W + b)
    post_act = activation(pre_act)

    

    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

# Q 2.2.2 
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    ans = np.zeros(x.shape)
    for i in range(x.shape[0]):
        c = np.max(x[i])
        ans[i] = np.exp(x[i] - c) / np.sum(np.exp(x[i] - c)) 

    return ans

# Q 2.2.3
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):

    #loss, acc = None, None
    count = 0
    loss = -np.sum(y * np.log(probs))
    #print (y[0], probs[0])
    for i in range (y.shape[0]):
        if (np.argmax(y[i]) == np.argmax(probs[i])):
            count += 1

    acc = count / y.shape[0]
    return loss, acc 

# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

# Q 2.3.1
def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]
    # your code here
    # do the derivative through activation first
    # then compute the derivative W,b, and X

    deriv = activation_deriv(post_act)
    grad_X = np.zeros(X.shape)
    grad_W = np.zeros(W.shape)

    loss_y = delta * deriv
    for i in range(X.shape[0]):
      grad_W += np.dot(X[i].reshape(-1, 1), loss_y[i].reshape(1, -1))
    
    grad_X = np.dot(delta, W.T)
    grad_b = np.sum(loss_y, axis=0)


    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X

# Q 2.4.1
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    batches = []
    shuffle = np.arange(x.shape[0])
    np.random.shuffle(shuffle)
    x = x[shuffle]
    y = y[shuffle]
    n = int(x.shape[0] / batch_size)
    x = np.array_split(x, n)
    y = np.array_split(y, n)
    for i in range(n):
      batches.append((x[i], y[i]))

    
    return batches
