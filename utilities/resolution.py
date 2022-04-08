from .jacobian import get_jacobian

import numpy as np

# In order to achieve a resolution of 1 on at least every triangle, we need this
#  resolution in the UV map
# (THIS IS UNSCALED!)
def get_resolution(v, f, uv, ftc):
    #Scale UV map with a constant so that it fits into the unit square.
    uv_dists = np.amax(uv, axis=0) - np.amin(uv, axis=0)
    uv = (uv-np.amin(uv, axis=0)) / max(uv_dists)

    #Each triangle gets stretched in its two directions by the singular values.
    #Rotations are irrelevant.
    J, _ = get_jacobian(v, f, uv, ftc)
    sigmas = np.linalg.svd(J)[1]

    #If the singular values are all 1, then a resolution of 1 can be achieved
    # with a resolution of 1 in the UV map.
    #If a triangle is scaled in any direction by sigma, then we need a
    # resolution of 1/sigma in the uv map to get a resolution of 1.
    #Thus, the resolution function is the max of the inverse of the singular
    # values (or Inf if there is a 0 singular value).
    
    return np.amax(1 / np.abs(sigmas))
