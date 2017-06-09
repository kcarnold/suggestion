import numpy as np


from scipy.spatial.distance import pdist, squareform

def scalar_dpp_diversity(x, max_distance=1.):
    x = np.array(x)[:,None]
    K = max_distance - squareform(pdist(x))
    K /= max_distance
    return np.linalg.det(K)

def scalar_mean_pdist_diversity(x):
    x = np.array(x)[:,None]
    return np.mean(pdist(x))

scalar_diversity = scalar_mean_pdist_diversity
# scalar_diversity = scalar_dpp_diversity
