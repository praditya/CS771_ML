import numpy as np
import random as rnd
import time as tm

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE SUBMIT.PY
# DO NOT INCLUDE PACKAGES LIKE SKLEARN, SCIPY, KERAS ETC IN YOUR CODE
# THE USE OF ANY MACHINE LEARNING LIBRARIES FOR WHATEVER REASON WILL RESULT IN A STRAIGHT ZERO
# THIS IS BECAUSE THESE PACKAGES CONTAIN SOLVERS WHICH MAKE THIS ASSIGNMENT TRIVIAL

# DO NOT CHANGE THE NAME OF THE METHOD "solver" BELOW. THIS ACTS AS THE MAIN METHOD AND
# WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THIS NAME WILL CAUSE EVALUATION FAILURES

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length
def stepLengthGenerator(mode, eta):
    if mode == "constant":
        return lambda t: eta
    elif mode == "linear":
        return lambda t: eta/(t+1)
    elif mode == "quadratic":
        return lambda t: eta/np.sqrt(t+1)

def normalGD(X, y, w):
    value=2*((X.T)@(X@w-y))
    return value

def Softhreshold(w,StepFunc,t):
    alpha = StepFunc(t)
    # alpha = stepLengthGenerator( "linear", eta )
    prox = np.zeros_like(w)
    for i in range(len(w)):
        if w[i] > alpha:
            prox[i] = w[i]-alpha
        elif w[i] < -alpha:
            prox[i] = w[i]+alpha
        else:
            prox[i] = 0
    return prox

def LassoGD(X, y, wHat):
    res = X.dot(wHat)-y
    GradL = (np.sign(wHat))+2*X.T.dot(res)
    return GradL

def DoCD(X,y,w,StepFunc,t):
    alpha = 1
    (n, d) = X.shape
    (n, d) = X.shape
    z  = np.sum(X**2,axis=0)
    # used 'a' for random permutation CD 
    a = np.random.permutation(d)
    for j in range(d):
        j = a[j]
        Xj = X[:,j]   
        res = y-X.dot(w)+w[j]*Xj
        pho = Xj.T.dot(res)

        if pho < -alpha/2:
            w[j] = (pho + alpha/2)/z[j]
        elif pho > alpha/2:
            w[j] = (pho - alpha/2)/z[j]
        else:
            w[j] = 0
    return w 

def DoGD(X,y,w,stepFunc,t):
    g = LassoGD(X, y, w)
    w = w-stepFunc(t)*g
    return w

def DoProxGD(X,y,w,StepFunc,t):
    g=normalGD(X,y,w)
    alpha = StepFunc(t)
    w_new=Softhreshold(w-alpha*g,StepFunc,t)
    return w_new


################################
# Non Editable Region Starting #
################################
def solver( X, y, timeout, spacing ):
	(n, d) = X.shape
	t = 0
	totTime = 0
	
	# w is the model vector and will get returned once timeout happens
	w = np.zeros( (d,) )
	tic = tm.perf_counter()
################################
#  Non Editable Region Ending  #
################################

	# You may reinitialize w to your liking here
	# You may also define new variables here e.g. step_length, mini-batch size etc
	eta = 0.089
	# eta = 0.12
	stepFunc = stepLengthGenerator( "constant", eta )
	# stepFunc = stepLengthGenerator( "quadratic", eta )
################################
# Non Editable Region Starting #
################################
	while True:
		t = t + 1
		if t % spacing == 0:
			toc = tm.perf_counter()
			totTime = totTime + (toc - tic)
			if totTime > timeout:
				return (w, totTime)
			else:
				tic = tm.perf_counter()
################################
#  Non Editable Region Ending  #
################################

		# Write all code to perform your method updates here within the infinite while loop
		# The infinite loop will terminate once timeout is reached
		# Do not try to bypass the timer check e.g. by using continue
		# It is very easy for us to detect such bypasses which will be strictly penalized
		
		w = DoProxGD(X,y,w,stepFunc,t)
		# w = DoGD(X,y,w,stepFunc,t)
		# w = DoGD(X,y,w,stepFunc,t)

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
		
	return (w, totTime) # This return statement will never be reached