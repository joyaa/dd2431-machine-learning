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
import sys
import math

from labfuns import *
from sklearn import decomposition
from matplotlib.colors import ColorConverter

import warnings
warnings.filterwarnings('ignore')

txt_file = 'accuracy.txt'


# ## Bayes classifier functions to implement
#
# The lab descriptions state what each function should do.

# Note that you do not need to handle the W argument for this part
# in: labels - N x 1 vector of class labels
# out: prior - C x 1 vector of class priors
def computePrior(labels, W=None):
    if W is None:
        labels = [int(label) for label in labels]
        return np.bincount(labels) / len(labels)

    prior = np.empty(len(set(labels)))
    for i in range(len(labels)):
        prior[labels[i]] += W[i]
    return prior


# Note that you do not need to handle the W argument for this part
# in:      X - N x d matrix of N data points
#     labels - N x 1 vector of class labels
# out:    mu - C x d matrix of class means
#      sigma - d x d x C matrix of class covariances
def mlParams(X, labels, W=None):
    c = len(set(labels))
    n = len(labels)
    d = len(X[0])
    mu = np.zeros([c, d])

    if W is None:
        W = np.ones(n)

    # print('W', W)
    for k in range(c):
        for i in range(d):
            mu_sum = 0
            n_sum = 0
            for ni in range(n):
                if labels[ni] == k:
                    mu_sum += X[ni][i] * W[ni]
                    n_sum += W[ni]
            mu[k][i] = mu_sum / n_sum

    sigma = np.zeros([d, d, c])
    for k in range(c):
        n_sum = 0
        for ni in range(n):
            if labels[ni] == k:
                n_sum += W[ni]
                subtracted = np.subtract(X[ni], mu[k])
                sigma[:, :, k] += multiply_matrix(subtracted) * W[ni]
        sigma[:, :, k] /= n_sum

    return mu, sigma


def multiply_matrix(subtracted):
    n = len(subtracted)
    matrix = np.empty([n, n])
    for i in range(n):
        for j in range(n):
            matrix[i][j] = subtracted[i] * subtracted[j]
    return matrix


# in:      X - N x d matrix of M data points
#      prior - C x 1 vector of class priors
#         mu - C x d matrix of class means
#      sigma - d x d x C matrix of class covariances
# out:     h - N x 1 class predictions for test points
def classify(X, prior, mu, sigma, covdiag=True):
    # Your code here
    # Example code for solving a psd system
    # L = np.linalg.cholesky(A)
    # y = np.linalg.solve(L,b)
    # x = np.linalg.solve(L.H,y)
    h = np.zeros(len(X))
    d = len(sigma)
    n = len(h)
    c = len(prior)

    if not covdiag:
        for ni in range(n):
            highest_delta = None
            for k in range(c):
                first_value = 0
                for i in range(d):
                    first_value += np.log(sigma[i][i][k])
                first_value = np.log(np.linalg.det(sigma[:, :, k])) / 2
                solved_equation = np.transpose(
                    [solve_equation(sigma[:, :, k], np.transpose(np.subtract(X[ni], mu[k])))])
                subtracted = np.transpose(np.transpose([np.subtract(X[ni], mu[k])]))
                second_value = np.dot(subtracted, solved_equation)[0][0]
                third_value = np.log(prior[k])
                delta = -first_value - second_value / 2 + third_value
                if highest_delta is None or delta > highest_delta:
                    highest_delta = delta
                    h[ni] = k
    else:
        for ni in range(n):
            highest_delta = None
            for k in range(c):
                diag = np.diag(np.diag(sigma[:, :, k]))
                first_value = np.log(np.linalg.det(diag)) / 2
                subtract = np.subtract(X[ni], mu[k])
                second_value = np.dot(np.dot(subtract, np.linalg.inv(diag)), np.transpose(subtract)) / 2
                third_value = np.log(prior[k])
                delta = -first_value - second_value / 2 + third_value
                if highest_delta is None or delta > highest_delta:
                    highest_delta = delta
                    h[ni] = k

    return h


def solve_equation(A, b):
    L = np.linalg.cholesky(A)
    y = np.linalg.solve(L, b)
    x = np.linalg.solve(np.transpose(L), y)
    return x


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
def trainBoost(X, labels, T=5, covdiag=True):
    c = len(set(labels))
    d = len(X[0])
    n = len(labels)

    priors = np.zeros([T, c])
    mus = np.zeros([T, c, d])
    sigmas = np.zeros([T, d, d, c])
    alphas = np.zeros(T)

    weights = np.ones(n) / n
    # print('weights', weights)
    for t in range(T):
        mu, sigma = mlParams(X, labels, weights)
        prior = computePrior(labels, weights)
        hi = classify(X, prior, mu, sigma, covdiag)
        priors[t] = prior
        mus[t] = mu
        sigmas[t] = sigma

        error_sum = 0
        for label_index, label in enumerate(labels):
            if hi[label_index] != label:
                error_sum += weights[label_index]

        # print('error_sum', error_sum)
        if error_sum == 0:
            error_sum = 0.0000001

        alphas[t] = (np.log(1 - error_sum) - np.log(error_sum)) / 2

        for label_index, label in enumerate(labels):
            if hi[label_index] == label:
                weights[label_index] *= math.exp(-alphas[t])
            else:
                weights[label_index] *= math.exp(alphas[t])
        weights = normalize(weights)

    return priors, mus, sigmas, alphas


def normalize(weights):
    sum = 0
    for weight in weights:
        sum += weight
    return weights / sum


# in:       X - N x d matrix of N data points
#      priors - length T list of prior as above
#         mus - length T list of mu as above
#      sigmas - length T list of sigma as above
#      alphas - T x 1 vector of vote weights
# out:  yPred - N x 1 class predictions for test points
def classifyBoost(X, priors, mus, sigmas, alphas, covdiag=True):
    n = len(X)
    T = len(alphas)
    c = len(priors[0])
    matrix = np.zeros([n, c])
    for t in range(T):
        ht = classify(X, priors[t], mus[t], sigmas[t], covdiag)
        for ni in range(n):
            matrix[ni][ht[ni]] += alphas[t]

    yPred = np.empty(n)
    for ni in range(n):
        likliest_class = None
        highest_vote_value = None
        for ci in range(c):
            if likliest_class is None or matrix[ni][ci] > highest_vote_value:
                likliest_class = ci
                highest_vote_value = matrix[ni][ci]
        yPred[ni] = likliest_class
    return yPred


# ## Define our testing function
#
# The function below, `testClassifier`, will be used to try out the different datasets.
# `fetchDataset` can be provided with any of the dataset arguments `wine`, `iris`, `olivetti` and `vowel`.
# Observe that we split the data into a **training** and a **testing** set.
def testClassifier(dataset='iris', dim=0, split=0.7, doboost=False, boostiter=5, covdiag=True, ntrials=10):
    X, y, pcadim = fetchDataset(dataset)

    means = np.zeros(ntrials, );

    for trial in range(ntrials):

        # xTr,yTr,xTe,yTe,trIdx,teIdx = trteSplit(X,y,split)
        xTr, yTr, xTe, yTe, trIdx, teIdx = trteSplitEven(X, y, split)

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
            priors, mus, sigmas, alphas = trainBoost(xTr, yTr, T=boostiter)
            yPr = classifyBoost(xTe, priors, mus, sigmas, alphas, covdiag=covdiag)
        else:
            ## Simple
            # Compute params
            prior = computePrior(yTr)
            mu, sigma = mlParams(xTr, yTr)
            # Predict
            yPr = classify(xTe, prior, mu, sigma, covdiag=covdiag)

        # Compute classification error
        print(
            "Trial:", trial, "Accuracy", 100 * np.mean((yPr == yTe).astype(float))
        )

        means[trial] = 100 * np.mean((yPr == yTe).astype(float))

    mean = np.mean(means)
    std_dev = np.std(means)
    print(
        "Final mean classification accuracy ", mean, "with standard deviation", std_dev
    )

    with open(txt_file, 'a') as f:
        f.write('dataset: ' + dataset + ', covdiag: ' + str(covdiag) + ', boost: ' + str(doboost) + '\n')
        f.write('Final mean classification accuracy: ' + str(mean) + '\n')
        f.write('Standard deviation: ' + str(std_dev) + '\n')
        f.write('\n')


# ## Plotting the decision boundary
#
# This is some code that you can use for plotting the decision boundary
# boundary in the last part of the lab.

def plotBoundary(dataset='iris', split=0.7, doboost=False, boostiter=5, covdiag=True):
    name = file_name(dataset, covdiag, doboost)
    plt.figure(name)
    X, y, pcadim = fetchDataset(dataset)
    xTr, yTr, xTe, yTe, trIdx, teIdx = trteSplitEven(X, y, split)
    pca = decomposition.PCA(n_components=2)
    pca.fit(xTr)
    xTr = pca.transform(xTr)
    xTe = pca.transform(xTe)

    pX = np.vstack((xTr, xTe))
    py = np.hstack((yTr, yTe))

    if doboost:
        ## Boosting
        # Compute params
        priors, mus, sigmas, alphas = trainBoost(xTr, yTr, T=boostiter, covdiag=covdiag)
    else:
        ## Simple
        # Compute params
        prior = computePrior(yTr)
        mu, sigma = mlParams(xTr, yTr)

    xRange = np.arange(np.min(pX[:, 0]), np.max(pX[:, 0]), np.abs(np.max(pX[:, 0]) - np.min(pX[:, 0])) / 100.0)
    yRange = np.arange(np.min(pX[:, 1]), np.max(pX[:, 1]), np.abs(np.max(pX[:, 1]) - np.min(pX[:, 1])) / 100.0)

    grid = np.zeros((yRange.size, xRange.size))

    for (xi, xx) in enumerate(xRange):
        for (yi, yy) in enumerate(yRange):
            if doboost:
                ## Boosting
                grid[yi, xi] = classifyBoost(np.matrix([[xx, yy]]), priors, mus, sigmas, alphas, covdiag=covdiag)
            else:
                ## Simple
                grid[yi, xi] = classify(np.matrix([[xx, yy]]), prior, mu, sigma, covdiag=covdiag)

    classes = range(np.min(y), np.max(y) + 1)
    ys = [i + xx + (i * xx) ** 2 for i in range(len(classes))]
    colormap = cm.rainbow(np.linspace(0, 1, len(ys)))

    # plt.hold(True)
    conv = ColorConverter()
    for (color, c) in zip(colormap, classes):
        try:
            CS = plt.contour(xRange, yRange, (grid == c).astype(float), 15, linewidths=0.25,
                             colors=conv.to_rgba_array(color))
        except ValueError:
            pass
        xc = pX[py == c, :]
        plt.scatter(xc[:, 0], xc[:, 1], marker='o', c=color, s=40, alpha=0.5)

    plt.xlim(np.min(pX[:, 0]), np.max(pX[:, 0]))
    plt.ylim(np.min(pX[:, 1]), np.max(pX[:, 1]))
    # plt.show()
    plt.savefig(name)


# ## Run some experiments
#
# Call the `testClassifier` and `plotBoundary` functions for this part.
# Example usage of the functions
def runExperiment(dataset, covdiag, doboost, split=0.7, boostiter=5):
    testClassifier(dataset=dataset, split=split, doboost=doboost, boostiter=boostiter, covdiag=covdiag)
    plotBoundary(dataset=dataset, split=split, doboost=doboost, boostiter=boostiter, covdiag=covdiag)


def setNumpyPrintOptions():
    np.set_printoptions(threshold=np.nan)
    np.set_printoptions(precision=25)
    np.set_printoptions(linewidth=200)


# ## Test the Maximum Likelihood estimates
#
# Call `genBlobs` and `plotGaussian` to verify your estimates.
def testEstimates():
    X, labels = genBlobs(centers=5)
    mu, sigma = mlParams(X, labels)
    plotGaussian(X, labels, mu, sigma)


def file_name(data_set, covdiag, doboost):
    return data_set + '_' + 'covdiag_' + str(covdiag).lower() + '_doboost_' + str(doboost).lower() + '.png'


def main():
    # testEstimates()
    setNumpyPrintOptions()
    data_sets = ('iris', 'wine', 'olivetti', 'vowel')

    for data_set in data_sets:
        runExperiment(data_set, False, False)
        runExperiment(data_set, False, True)
        runExperiment(data_set, True, False)
        runExperiment(data_set, True, True)

if __name__ == '__main__':
    main()