import numpy as np

def Covariance(m):
    """
    Computes the covariance matrix given a matrix

    Parameters: m: a numpy array
    

    Returns: A numpy covariance array
    """
    mean = np.mean(m, axis=1)

    for i in range(0, m.shape[1]):
            if i == 0:
                S = (m[:,i] - mean) * np.transpose((m[:,i] - mean)) ## Covariance Matrix
            else:
                S += (m[:,i] - mean) * np.transpose((m[:,i] - mean))

    S /= m.shape[1]
    return S

def PCD(m, n):
    """
    Computes the principal component decomposition of the input matrix

    Parameters: m: numpy array to be decomposed

                n: number of eigenvectors

    Returns: A numpy array of Eigenvectors and a python list of Eigenvalues
    """
    mean = np.mean(m, axis=1)
    if n > m.shape[1]:
        pass
    else:
        C = np.dot(np.transpose(m), m)
        eig = np.linalg.eigh(C)
