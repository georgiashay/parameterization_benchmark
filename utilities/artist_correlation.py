import numpy as np

def get_artist_correlation(singular_values_artist, singular_values_measure, mesh_areas):
    #Account for numerically small singular values
    eps = np.finfo(singular_values_measure.dtype).eps
    singular_values_artist = np.maximum(singular_values_artist, eps)
    singular_values_measure = np.maximum(singular_values_measure, eps)

    #Log
    s1 = np.log(singular_values_artist)
    s2 = np.log(singular_values_measure)

    #Area weight
    w = np.tile(mesh_areas.T, (1,2))
    w /= np.sum(w)

    #Compute weighted correlation
    def cov(X, Y, w):
        EX = np.average(X, weights=w)
        EY = np.average(Y, weights=w)
        return np.average((X-EX)*(Y-EY), weights=w)
    def std(X, w):
        return np.sqrt(cov(X, X, w))
    rho = cov(s1, s2, w) / (std(s1,w)*std(s2,w))

    return np.abs(rho - 1.)

