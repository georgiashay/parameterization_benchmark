import igl
import numpy as np

def get_uv_boundary_length(uv, ftc):
    boundary_uv_edges = uv[igl.boundary_facets(ftc)]
    boundary_length = np.sum(np.linalg.norm(boundary_uv_edges[:,0,:] - boundary_uv_edges[:,1,:], axis=1))
    
    #TODO: EXCLUDE ORIGINAL BOUNDARY
    new_boundary_length = boundary_length
    
    return boundary_length, new_boundary_length