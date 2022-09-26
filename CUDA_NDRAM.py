# Rey Gonzalez
# NDRAM Implementation (Independent Project)
# AKRaNLU@Purdue, Heavilon 108

import numpy as np
import time
import numba as cuda

# learning rule from the paper
# inputs:
## W <- nXn weight matrix
## x0 <- n-shaped stimulus array
## xt <- n-shaped transformed stimulus array
## h <- learning parameter
# output: updated weight matrix <- W
@cuda.jit
def ndram_learn(W, x0, xt, h):
    return W + h * (np.outer(x0, x0) - np.outer(xt, xt))

# convergence equation (convergence: _lambda -> 1; d_delta -> 0)
# inputs:
## h <- learning parameter
## delta <- transmission parameter
## _lambda <- upper eigenvalue
@cuda.jit
def delta_lambda(h, delta, _lambda):
    return h * (1 - np.square((delta + 1) * _lambda - delta * _lambda**3))

# non-linear transmission rule
# inputs:
## ai <- element to be transmitted
## delta <- transmission parameter
# output: transmission amount
# (feels unnatural to send logic to GPU, but worth a shot!)
@cuda.jit
def transmission_single(ai, delta):
    if ai > 1:
        return 1
    elif ai < -1:
        return -1
    return ((delta + 1) * ai - delta * ai**3)

# list-based transmission helper
# inputs:
## a <- list of values for (re-)transmission
## delta <- transmission parameter
# output: transmitted list
def transmission(a, delta):
    return [transmission_single(ai, delta) for ai in a]

# transmission helper with n/iterations
# inputs:
## W <- weight matrix
## x <- signal for transmission
## delta <- transmission parameter
## n <- number of transmissions
# output: transmitted list <- xt
def transmission_n(W, x, delta, n):
    xt = x
    for i in range(n):
        a = np.dot(W, xt)
        xt = transmission(a, delta)
    return xt

# transmission n, learn once
# inputs:
## W <- weight matrix
## x <- signal for transmission
## delta <- trabsmission parameter
## h <- learning parameter
## n <- number of transmissions
# output: updated weight matrix <- W
def transmit_and_learn(W, x, delta, h, n):
    xt = transmission_n(W, x, delta, n)
    return ndram_learn(W, x, xt, h)

# generate initial weight matrix
# input: length of side
# output: nXn matrix
def initial_weights(n):
    return np.zeros([n,n])

# output top eigenvalue and the delta
# inputs:
## W <- weight matrix
## h <- learning parameter
## delta <- transmission parameter
# outputs:
## top eigenvalue
## delta in eigenvalue
def convergence(W, h, delta):
    _lambda = np.linalg.eigvals(W)[0]
    d_lambda = delta_lambda(h, delta, _lambda)

    return _lambda, d_lambda

# print updates
def progress(timer, counter, _lambda, d_lambda):
    timer2 = time.time()
    if counter%10 == 0:
        print(counter, ": total elapsed: ", timer2-timer, ", lambda: ", _lambda, ", d_lambda: ", d_lambda)
    counter+=1
    return counter

def loop_init():
    return 0, 1, 1, time.time()

# test - 28/3 signals with subsequent 1s (paper experiments w/ 34% load)
side = 128
blank = [-1] * side
blank = np.array(blank)
stimuli = [blank.copy() for i in range(int(side/3))]

for i, x0 in enumerate(stimuli):
    x0[i] = 1

# generate weights
W = initial_weights(side)

# learning parameter (suggested 0.001 - 0.00197)
h = 0.00197
# transmission parameter (suggested 0.1 - 0.5)
delta = 0.5

# prepare data
_lambda, d_lambda, counter, timer = loop_init()

# easy to loop around _lambda (converges to 1)
while _lambda < 0.999:
    
    # for each stimuli, transmit n times, then update weight matrix
    for x0 in stimuli:
        W = transmit_and_learn(W, x0, delta, h, 10)

    # update convergence values
    _lambda, d_lambda = convergence(W, h, delta)

    # readout
    counter = progress(timer, counter, _lambda, d_lambda)

# sanity check with the orginal stimuli
for x0 in stimuli:
    print(np.dot(W, x0))