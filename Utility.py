
import numpy as np


def PCA(m, n):
    """
    Computes the principal component analysis of the input matrix

    Parameters: m: numpy array to be decomposed

                n: number of components

    Returns: A numpy array of Eigenvectors and a python list of Eigenvalues
    """
    mean = np.mean(m, axis=1)
    m = m - np.tile(mean, (n, 1)).T
    if m.shape[1] <= m.shape[0]:
        C = np.dot(np.transpose(m), m)
        eigvals, eigvectors = np.linalg.eigh(C)
        eigvectors = np.dot(m, eigvectors)
        for i in range(0, m.shape[1]):
            eigvectors[:,i] = eigvectors[:,i]/np.linalg.norm(eigvectors[:,i])
    else:
        C = np.dot(m, np.transpose(m))
        eigvals, eigvectors = np.linalg.eigh(C)
        
    eigvectors = eigvectors[:,np.argsort(eigvals)]
    return eigvals, eigvectors, mean

def Normalize(m, low=0, high=255):
    """
    Normalizes a numpy array
    """
    min, max = np.min(m), np.max(m)
    m = m - min
    m = m / (max - min)
    m = m * (high - low)
    m = m + low
    return m

def Project(x, w, mean=0):
    scalars = np.dot((x-mean).T, w)
    #return scalars
    proj = np.zeros(shape=(x.shape[0]))
    for i in xrange(w.shape[1]):
        proj = proj + scalars[i] * w[:,i]
    return proj
