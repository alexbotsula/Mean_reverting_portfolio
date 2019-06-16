# Utility function defined in 'Mean Reversion Optimisation Part1.ipynb'

# Imports
import matplotlib.pyplot as plt
import numpy as np

import cvxpy as cp
from sklearn.decomposition import SparsePCA
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# Build autocovariance matrix
def autocovariance_matrix(x, k=0):
    T = x.shape[0]
    x_tilde = x - np.nanmean(x, axis=0)
    
    if k > 0:
        A_k = 1/(T - k - 1) * np.dot(x_tilde[k:,:].transpose(), x_tilde[:-k])
    else:
        A_k = 1/(T - 1) * np.dot(x_tilde.transpose(), x_tilde)
    return A_k

# Plot time series with basic stats
def plot_mr_ts(ts, label):
    var_ = np.var(ts)
    adf_ = adfuller(ts)
    plt.plot(ts, label='{}; Var:{:.0f} ADF p:{:.3f}'.format(label, var_, adf_[1]))
    plt.ylabel('Close')
    plt.legend()
    plt.show()

# Search for Sparse PCA alpha parameter that provides no more than required number of features in sparse maximum principal component
def spca_alpha(Y_matrix, max_features, eps = 1e-4):
    """
    Y_matrix = input matrix
    max_features = maximum number of non-zero elements in the first sparse principal component
    eps = convergence parameter
    """
    range_ = [0,1]

    while True:
        range_mid = 0.5 * (range_[1] + range_[0])

        sp_pca_mid = SparsePCA(n_components=1, alpha=range_mid)
        sp_pca_mid.fit(Y_matrix)
        n_features_mid = sum([x != 0 for x in sp_pca_mid.components_[0]])

        if n_features_mid == max_features:
            return range_mid
        elif range_[1] - range_[0] < eps:
            return None
        elif n_features_mid < max_features:
            range_[1] = range_mid
        else:
            range_[0] = range_mid

# Solve PSD problem for Portmanteau criterion
def portmanteau_matrix(X, rho_port, nu_port, n_a, A0):  
    """
    X = source data
    rho_port = weight of L2 norm penalty
    nu_port = minimum required variance
    n_a = number of autoregressive components to consider
    A0 = zero lag covariance matrix
    """
    Y_port = cp.Variable(A0.shape, PSD=True)

    target_port = rho_port * cp.norm(Y_port, 2)
    for i in range(1, n_a + 1):
        A_i = autocovariance_matrix(X, i)
        target_port += cp.square(cp.trace(cp.matmul(A_i, Y_port)))

    constraints_port = [Y_port >> 0]
    constraints_port += [cp.trace(cp.matmul(A0, Y_port)) >= nu_port]
    constraints_port += [cp.trace(Y_port) == 1]

    prob_port = cp.Problem(cp.Minimize(target_port), 
                           constraints_port)

    res_solve_port = prob_port.solve(verbose=False, max_iters=int(5e6))
    print('Optimization target:\t{:.3f}'.format(res_solve_port))
    return Y_port.value

