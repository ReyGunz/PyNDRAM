# Rey Gonzalez
# NDRAM Implementation (Independent Project)
# AKRaNLU@Purdue, Heavilon 108

import numpy as np
import time

# learning rule from the paper
# inputs:
## W <- nXn weight matrix
## x0 <- n-shaped stimulus array
## xt <- n-shaped transformed stimulus array
## h <- learning parameter
# output: updated weight matrix <- W
def ndram_learn(W, x0, xt, h):
    left = np.outer(x0, x0)
    right = np.outer(xt, xt)

    return np.array(W + h * np.array(left - right))

# convergence equation (convergence: _lambda -> 1; d_delta -> 0)
# inputs:
## h <- learning parameter
## delta <- transmission parameter
## _lambda <- upper eigenvalue
def delta_lambda(h, delta, _lambda):
    return h * (1 - np.square((delta + 1) * _lambda - delta * _lambda**3))

# non-linear transmission rule
# inputs:
## ai <- element to be transmitted
## delta <- transmission parameter
# output: transmission amount
def transmission_single(ai, delta):
    if ai > 1:
        return 1
    elif ai < -1:
        return -1
    else:
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
def readout_and_counter(timer, counter, _lambda, d_lambda):
    timer2 = time.time()
    if counter%5 == 0:
        print(counter, ": total elapsed: ", time.time()-timer, ", lambda: ", _lambda, ", d_lambda: ", d_lambda)
    counter+=1
    return counter

def loop_init():
    return 0, 1, 1, time.time()

# test - 28 signals with subsequent 1s
side = 28
blank = [-1] * side
blank = np.array(blank)
stimuli = [blank.copy() for i in range(28)]

for i, x0 in enumerate(stimuli):
    x0[i] = 1

# generate weights, provide parameters (as suggested in NDRAM paper)
W = initial_weights(side)
h = 0.001
delta = 0.1

# prepare data
_lambda, d_lambda, counter, timer = loop_init()

# easy to loop around _lambda
while _lambda < 0.999:
    counter = readout_and_counter(timer, counter, _lambda, d_lambda)
    
    # for each stimuli, transmit n times, then learn
    for x0 in stimuli:
        W = transmit_and_learn(W, x0, delta, h, 3)

    # update convergence values
    _lambda, d_lambda = convergence(W, h, delta)

# sanity check with the orginal stimuli
for x0 in stimuli:
    print(np.dot(W, x0))