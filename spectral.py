# Parameters:
# -----------
# X: 
#     shape(n_samples, n_features)
#     Data from which to compute the covariance estimate
# rho: 
#     float>0 
#     reglularization factor
# assume_centered: 
#     Boolen
#     If True, data are not  centered before computation
#     If Talse, data are centered befire compututation
# Returns:
# -----------
# est_cov:
#     shape(n_features, n_features)
#     covariance
# precision:
#     shape(n_features, n_features)
#     inverse convarian matrix


from scipy import linalg
import math
import numpy as np


def spectral(X, rho, assume_centered=False):
   N, d = X.shape
   # mu = data.mean(1)
   if not assume_centered:
      X = X - X.mean(axis=0, keepdims=True)

   if N <= d:
      # dimension high than sample size
      U, D, Vh = linalg.svd(X, full_matrices=False)
      # Vh, D_tile, U = linalg.svd(X, full_matrices=False)
      D_tilde = np.diag(D) ** 2 / N
      sample_cov = np.dot(U, np.dot(D_tilde, U.T))
      dRic = np.zeros((d, d))
      for i in xrange(d):
         dRic[i, i] = math.sqrt(1.0 / rho + D_tilde[i, i] ** 2 / 4*(rho ** 2)) - D_tilde[i, i] / (2*rho) - 1.0 / math.sqrt(rho)

      precision = np.dot(U, np.dot(dRic, U.T)) + np.identity(d) / math.sqrt(rho)
   else:
      # sample size higher than dimension, SVD on sample covariance
      sample_cov = np.dot(X.T, X)/float(N)
      U, D, Vh = linalg.svd(sample_cov, full_matrices=True)
      dRic = np.zeros((d, d))
      for i in xrange(d):
         dRic[i, i] = math.sqrt(1.0 / rho + D[i] ** 2 / 4*(rho ** 2)) - D[i] / (2*rho) - 1.0 / math.sqrt(rho)
      precision = np.dot(U, np.dot(dRic, U.T)) + np.identity(d) / math.sqrt(rho)

   return sample_cov, precision
