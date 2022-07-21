
import numpy as np

# Input the singular values of the scaled mesh as well as the scaled areas to
# compute the symmetric Dirichlet energy.
def get_symmetric_dirichlet_energy(singular_values, areas):
    s2 = singular_values ** 2.
    np.seterr(divide='ignore')
    inv_s2 = np.where(s2 != 0, 1./s2, np.nan)
    np.seterr(divide='print')
    return np.sum(0.5*areas.T*(s2 + inv_s2))
