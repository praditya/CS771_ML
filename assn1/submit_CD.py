import numpy as np
import random as rnd
import time as tm

from matplotlib import pyplot as plt
# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE SUBMIT.PY
# DO NOT INCLUDE PACKAGES LIKE SKLEARN, SCIPY, KERAS ETC IN YOUR CODE
# THE USE OF ANY MACHINE LEARNING LIBRARIES FOR WHATEVER REASON WILL RESULT IN A STRAIGHT ZERO
# THIS IS BECAUSE THESE PACKAGES CONTAIN SOLVERS WHICH MAKE THIS ASSIGNMENT TRIVIAL

# DO NOT CHANGE THE NAME OF THE METHOD "solver" BELOW. THIS ACTS AS THE MAIN METHOD AND
# WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THIS NAME WILL CAUSE EVALUATION FAILURES


def stepLengthGenerator(mode, eta):
    if mode == "constant":
        return lambda t: eta
    elif mode == "linear":
        return lambda t: eta/(t+1)
    elif mode == "quadratic":
        return lambda t: eta/np.sqrt(t+1)

# For cyclic mode, the state is a tuple of the current coordinate and the number of dimensions


def getCyclicCoord(state):
    curr = state[0]
    d = state[1]
    if curr >= d - 1 or curr < 0:
        curr = 0
    else:
        curr += 1
    state = (curr, d)
    return (curr, state)

# For random mode, the state is the number of dimensions


def getRandCoord(state):
    d = state
    curr = random.randint(0, d - 1)
    state = d
    return (curr, state)

# For randperm mode, the state is a tuple of the random permutation and the current index within that permutation


def getRandpermCoord(state):
    idx = state[0]
    perm = state[1]
    d = len(perm)
    if idx >= d - 1 or idx < 0:
        idx = 0
        perm = np.random.permutation(d)
    else:
        idx += 1
    state = (idx, perm)
    curr = perm[idx]
    return (curr, state)

# Get functions that offer various coordinate selection schemes


def coordinateGenerator(mode, d):
    if mode == "cyclic":
        return (getCyclicCoord, (0, d))
    elif mode == "random":
        return (getRandCoord, d)
    elif mode == "randperm":
        return (getRandpermCoord, (0, np.random.permutation(d)))


# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length
def LassoGD(X, y, wHat):
    res = X.dot(wHat)-y
    GradL = (np.sign(wHat))+2*X.T.dot(res)
    return GradL

def Softhreshold(w):
    alpha = 1
    for i in range(len(w)):
        if w[i] > alpha:
            w[i] = w[i]-alpha
        elif w[i] < -alpha:
            w[i]=w[i]+alpha
        else:
            w[i]=0
    return w

def getObjValue(X, y, wHat):
    lassoLoss = np.linalg.norm(wHat, 1) + pow(np.linalg.norm(X.dot(wHat) - y, 2), 2)
    return lassoLoss
################################
# Non Editable Region Starting #
################################


def solver(X, y, timeout, spacing):
    (n, d) = X.shape
    t = 0
    totTime = 0

    # w is the model vector and will get returned once timeout happens
    w = 1.5*np.ones((d,))
    tic = tm.perf_counter()
################################
#  Non Editable Region Ending  #
################################
    # You may reinitialize w to your liking here
    # You may also define new variables here e.g. step_length, mini-batch size etc

    eta = 5e-3
    B = 100
    stepFunc = stepLengthGenerator( "constant", eta )
    # coordinateGenerator(mode, d)

    # w = np.ones((d,))
    objValseries = []
################################
# Non Editable Region Starting #
################################
    while True:
        t = t + 1
        if t % spacing == 0:
            toc = tm.perf_counter()
            totTime = totTime + (toc - tic)
            if totTime > timeout:
                return (w, totTime, objValseries)
            else:
                tic = tm.perf_counter()
################################
#  Non Editable Region Ending  #
################################
        g = LassoGD(X, y, w)
        # u = w-stepFunc(t)*g
        Sl = Softhreshold(w)
        res = X.dot(w)-y
        wp = w-stepFunc(t)*X.T.dot(res)
        w = Sl*(wp)
        # v = np.sign(w)
        # w = w-stepFunc(t)*g
        # w = (np.sign(u))*(np.max(abs(u),0))
        # w = u - v
        objValseries = np.append(objValseries,getObjValue(X,y,w))
    # Write all code to perform your method updates here within the infinite while loop
    # The infinite loop will terminate once timeout is reached
    # Do not try to bypass the timer check e.g. by using continue
    # It is very easy for us to detect such bypasses which will be strictly penalized

    # Please note that once timeout is reached, the code will simply return w
    # Thus, if you wish to return the average model (as is sometimes done for GD),
    # you need to make sure that w stores the average at all times
    # One way to do so is to define a "running" variable w_run
    # Make all GD updates to w_run e.g. w_run = w_run - step * delw
    # Then use a running average formula to update w
    # w = (w * (t-1) + w_run)/t
    # This way, w will always store the average and can be returned at any time
    # In this scheme, w plays the role of the "cumulative" variable in the course module optLib
    # w_run on the other hand, plays the role of the "theta" variable in the course module optLib

    return (w, totTime, objValseries)  # This return statement will never be reached

traindata = np.loadtxt( "train" )
# wAst = np.loadtxt( "wAstTest" )
k = 20
# objValseries = []
y = traindata[:,0]
X = traindata[:,1:]
(w,totTime,objValseries)=solver(X,y,10,10)
print (w)
wsparse_idx = np.argsort( np.abs(w) )[::-1][:20]
print (wsparse_idx)
norm1 = np.linalg.norm(w,1)
print (norm1)
plt.plot(objValseries)
plt.show()
# print (np.sign(w))
