
# coding: utf-8

# # Lab 3: Bayes Classifier and Boosting

# ## Jupyter notebooks
#
# In this lab, you can use Jupyter <https://jupyter.org/> to get a nice layout of your code and plots in one document. However, you may also use Python as usual, without Jupyter.
#
# If you have Python and pip, you can install Jupyter with `sudo pip install jupyter`. Otherwise you can follow the instruction on <http://jupyter.readthedocs.org/en/latest/install.html>.
#
# And that is everything you need! Now use a terminal to go into the folder with the provided lab files. Then run `jupyter notebook` to start a session in that folder. Click `lab3.ipynb` in the browser window that appeared to start this very notebook. You should click on the cells in order and either press `ctrl+enter` or `run cell` in the toolbar above to evaluate all the expressions.

# ## Import the libraries
#
# Check out `labfuns.py` if you are interested in the details.

import numpy as np
from scipy import misc
from imp import reload
from labfuns import *
from sklearn import decomposition
from matplotlib.colors import ColorConverter

import warnings
warnings.filterwarnings('ignore')

# ## Bayes classifier functions to implement
#
# The lab descriptions state what each function should do.

# Note that you do not need to handle the W argument for this part
# in: labels - N x 1 vector of class labels
# out: prior - C x 1 vector of class priors
def computePrior(labels,W=None):
    # Your code here
    if W == None:
        W=np.ones(len(labels))


    #kolla senare
    prior=np.zeros(len(set(labels)))

    for k in range(labels.size):
        prior[int(labels[k])]+=W[k]

    prior /= np.sum(W[k])

            
    return prior

# Note that you do not need to handle the W argument for this part
# in:      X - N x d matrix of N data points
#     labels - N x 1 vector of class labels
# out:    mu - C x d matrix of class means
#      sigma - d x d x C matrix of class covariances
def mlParams(X,labels,W=None):  #3.4 Assignment 1
    i = 0
    d = X[0].size
    classes=set(labels)
    C = len(classes)
    if W == None:
        W=np.ones(len(labels))
    mu = np.zeros([C,d], dtype=object)

    #nytt försök
    for k in range(C):
        for i in range(d):

            mu_k = 0
            n_k = 0.0
            for x in range(labels.size):
                #W[x] added to n_k and mu_k
                if labels[x] == k:
                    mu_k += X[x][i]*W[x]
                    n_k += W[x]             
            mu[k][i] = mu_k/n_k


    sigma = np.zeros([d,d,C])

    for k in range(C):
        n_k=0
        for x in range(labels.size):
            if labels[x]==k:
                #W[x] added to n_k and sigma
                n_k += W[x]
                diff = np.subtract(X[x],mu[k])
                sigma[:,:,k] = sigma[:,:,k]+W[x]*np.outer(diff,diff)
                #for d1 in rangecl(diff.size):
                #    for d2 in range(diff.size):
                #        sigma[d1][d2][k] += diff[d1]*diff[d2]
        
        sigma[:,:,k] /= n_k 
    #print np.shape(sigma)
    #sigma
    return mu, sigma

# in:      X - N x d matrix of M data points
#      prior - C x 1 vector of class priors
#         mu - C x d matrix of class means
#      sigma - d x d x C matrix of class covariances
# out:     h - N x 1 class predictions for test points
def classify(X,prior,mu,sigma,covdiag=True):
    # Your code here

    C = mu[:,0].size
    N = X[:,0].size
    delta=np.zeros(C)
    h=np.zeros(N)
    d = len(sigma[:,:,0])
    
    if covdiag:
        for n in range(N):
            maxDelta = 0
            maxClass = 0
            for k in range(C):  
                diff=X[n]-mu[k]      
                x = np.linalg.inv(np.diag(np.diag(sigma[:,:,k])))
                lnDetSigma = np.log(np.linalg.det(sigma[:,:,k]))   
                #print lnDetSigma,(np.transpose(diff)*x*diff) 
                delta[k]=-0.5*lnDetSigma-0.5*np.dot(np.dot(diff,x),np.transpose(diff))+np.log(prior[k])
                if delta[k]>maxDelta:
                    maxDelta = delta[k]
                    maxClass = k
            h[n] = maxClass#np.where(delta,md) 

    else:
        for n in range(N):
        # Example code for solving a psd system
            maxDelta = 0
            maxClass = 0
            for k in range(C):
                L = np.linalg.cholesky(sigma[:,:,k])
                y = np.linalg.solve(L,np.transpose(X[n]-mu[k]))
                x = np.linalg.solve(np.transpose(L),y)

                lnDetSigma = 2*np.sum(np.log(np.diag(L)))

                delta[k]=-0.5*lnDetSigma-0.5*np.dot((X[n]-mu[k]),x)+np.log(prior[k])
                if delta[k]>maxDelta:
                    maxDelta = delta[k]
                    maxClass = k
            h[n] = maxClass
            #h[n] = np.where(delta,max(delta)) 
    #print h 
    return h


# ## Test the Maximum Likelihood estimates
#
# Call `genBlobs` and `plotGaussian` to verify your estimates.

X, labels = genBlobs(centers=5)
W=np.ones(len(labels))/len(labels)
mu, sigma = mlParams(X,labels,W)

#plotGaussian(X,labels,mu,sigma)


# ## Boosting functions to implement
#
# The lab descriptions state what each function should do.

# in:       X - N x d matrix of N data points
#      labels - N x 1 vector of class labels
#           T - number of boosting iterations
# out: priors - length T list of prior as above
#         mus - length T list of mu as above
#      sigmas - length T list of sigma as above
#      alphas - T x 1 vector of vote weights
def trainBoost(X,labels,T=5,covdiag=True):
    # Your code here
    C =len(set(labels))
    d = len(X[0])
    weights = np.ones(len(labels))/len(labels)
    
    priors = np.zeros([T,C])
    mus = np.zeros([T,C,d])
    sigmas = np.zeros([T,d,d,C])
    alphas = np.zeros([T,C])
    for i in range(T):
        delta = np.zeros(len(labels))
        prior = computePrior(labels,weights)
        mu, sigma = mlParams(X,labels,weights)
        h = classify(X,prior,mu,sigma,covdiag)
        for j in range(len(labels)):
            if h[j] == labels[j]:
                delta[j]=1
  
        e=np.sum(np.dot(weights,(1-delta)))
        alpha = 0.5*(np.log(1-e)-np.log(e))
        for k in range(len(labels)):
            if delta[k] == 1:
                weights[k] *= np.exp(-alpha)
            else:
                weights[k] *= np.exp(alpha)
        weights/=np.sum(weights)        
        priors[i] = prior
        mus[i] = mu
        sigmas[i] = sigma
        alphas[i] = alpha

    #print np.shape(mus[0])

    return priors,mus,sigmas,alphas

# in:       X - N x d matrix of N data points
#      priors - length T list of prior as above
#         mus - length T list of mu as above
#      sigmas - length T list of sigma as above
#      alphas - T x 1 vector of vote weights
# out:  yPred - N x 1 class predictions for test points
def classifyBoost(X,priors,mus,sigmas,alphas,covdiag=True):
    # Your code here
   
    #c=np.zeros(len(mus))
    H=np.zeros(len(X))
    #print np.shape(alphas)
    C=len(mus[:,0]) #number of classes
    for i in range(len(X)):
        bestC=0
        votes = np.zeros(C)
        #c=np.zeros(len(mus))
        for j in range(C):
            for k in range(len(mus)):  #T hypothesis
                h = classify(np.atleast_2d(X[i]),priors[k],mus[k],sigmas[k],covdiag)
                if j == h:
                    votes[j]+=alphas[k][j]
            if votes[j] == max(votes): #segt med max varje gång
                bestC = j

        H[i] = bestC
    return H


# ## Define our testing function
#
# The function below, `testClassifier`, will be used to try out the different datasets. `fetchDataset` can be provided with any of the dataset arguments `wine`, `iris`, `olivetti` and `vowel`. Observe that we split the data into a **training** and a **testing** set.

np.set_printoptions(threshold=np.nan)
np.set_printoptions(precision=25)
np.set_printoptions(linewidth=200)

def testClassifier(dataset='iris',dim=0,split=0.7,doboost=False,boostiter=5,covdiag=True,ntrials=20):

    X,y,pcadim = fetchDataset(dataset)

    means = np.zeros(ntrials,);

    for trial in range(ntrials):

        # xTr,yTr,xTe,yTe,trIdx,teIdx = trteSplit(X,y,split)
        xTr,yTr,xTe,yTe,trIdx,teIdx = trteSplitEven(X,y,split)

        # Do PCA replace default value if user provides it
        if dim > 0:
            pcadim = dim
        if pcadim > 0:
            pca = decomposition.PCA(n_components=pcadim)
            pca.fit(xTr)
            xTr = pca.transform(xTr)
            xTe = pca.transform(xTe)

        ## Boosting
        if doboost:
            # Compute params
            priors,mus,sigmas,alphas = trainBoost(xTr,yTr,T=boostiter)
            yPr = classifyBoost(xTe,priors,mus,sigmas,alphas,covdiag=covdiag)
        else:
        ## Simple
            # Compute params 

            prior = computePrior(yTr)
            mu, sigma = mlParams(xTr,yTr)#Added W for boosting
            # Predict
            yPr = classify(xTe,prior,mu,sigma,covdiag=covdiag)

        # Compute classification error
        print "Trial:",trial,"Accuracy",100*np.mean((yPr==yTe).astype(float))

        means[trial] = 100*np.mean((yPr==yTe).astype(float))
    global results 
    results= "Final mean classification accuracy ", np.mean(means), "with standard deviation", np.std(means)
    print "Final mean classification accuracy ", np.mean(means), "with standard deviation", np.std(means)


# ## Plotting the decision boundary
#
# This is some code that you can use for plotting the decision boundary
# boundary in the last part of the lab.

def plotBoundary(dataset='iris',split=0.7,doboost=False,boostiter=5,covdiag=True):

    X,y,pcadim = fetchDataset(dataset)
    xTr,yTr,xTe,yTe,trIdx,teIdx = trteSplitEven(X,y,split)
    pca = decomposition.PCA(n_components=2)
    pca.fit(xTr)
    xTr = pca.transform(xTr)
    xTe = pca.transform(xTe)

    pX = np.vstack((xTr, xTe))
    py = np.hstack((yTr, yTe))

    if doboost:
        ## Boosting
        # Compute params
        priors,mus,sigmas,alphas = trainBoost(xTr,yTr,T=boostiter,covdiag=covdiag)
    else:
        ## Simple
        # Compute params
        
        prior = computePrior(yTr)
        mu, sigma = mlParams(xTr,yTr) #Added W for boosting

    xRange = np.arange(np.min(pX[:,0]),np.max(pX[:,0]),np.abs(np.max(pX[:,0])-np.min(pX[:,0]))/100.0)
    yRange = np.arange(np.min(pX[:,1]),np.max(pX[:,1]),np.abs(np.max(pX[:,1])-np.min(pX[:,1]))/100.0)

    grid = np.zeros((yRange.size, xRange.size))

    for (xi, xx) in enumerate(xRange):
        for (yi, yy) in enumerate(yRange):
            if doboost:
                ## Boosting
                grid[yi,xi] = classifyBoost(np.matrix([[xx, yy]]),priors,mus,sigmas,alphas,covdiag=covdiag)
            else:
                ## Simple
                grid[yi,xi] = classify(np.matrix([[xx, yy]]),prior,mu,sigma,covdiag=covdiag)

    classes = range(np.min(y), np.max(y)+1)
    ys = [i+xx+(i*xx)**2 for i in range(len(classes))]
    colormap = cm.rainbow(np.linspace(0, 1, len(ys)))

    plt.hold(True)
    conv = ColorConverter()
    for (color, c) in zip(colormap, classes):
        try:
            CS = plt.contour(xRange,yRange,(grid==c).astype(float),15,linewidths=0.25,colors=conv.to_rgba_array(color))
        except ValueError:
            pass   
        xc = pX[py == c, :]
        plt.scatter(xc[:,0],xc[:,1],marker='o',c=color,s=40,alpha=0.5)

    plt.xlim(np.min(pX[:,0]),np.max(pX[:,0]))
    plt.ylim(np.min(pX[:,1]),np.max(pX[:,1]))
    plt.show()


# ## Run some experiments
#
# Call the `testClassifier` and `plotBoundary` functions for this part.

# Example usage of the functions
#dataset = raw_input("data,covdiag?: ")
#covdiag = raw_input("covdiag?: ")
#testClassifier(dataset=dataset,split=0.7,doboost=True,boostiter=5,covdiag=covdiag)
#plotBoundary(dataset=dataset,split=0.7,doboost=True,boostiter=5,covdiag=covdiag)
testClassifier(dataset='vowel',split=0.7,doboost=False,boostiter=5,covdiag=False)
plotBoundary(dataset='vowel',split=0.7,doboost=False,boostiter=5,covdiag=False)
print results