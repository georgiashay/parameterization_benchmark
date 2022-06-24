
import numpy as np

# Input the singular values of the scaled mesh as well as the scaled areas to
# compute the symmetric Dirichlet energy.
def symmetric_dirichlet_energy(singular_values, areas):
    return np.sum(0.5*areas[:,None]*(singular_values**2. + 1./singular_values**2))
