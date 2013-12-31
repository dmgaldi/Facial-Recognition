import numpy as np

def Covariance(m):
    """
    Computes the covariance matrix given a matrix

    Parameters: m: a numpy array

    Returns: A numpy covariance array
    """
    mean = np.mean(trainingMatrix, axis=1)
