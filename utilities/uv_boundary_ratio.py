def get_uv_boundary_length(uv, ftc):
    boundary_uv_edges = uv[igl.boundary_facets(ftc)]
    boundary_length = np.sum(np.linalg.norm(boundary_uv_edges[:,0,:] - boundary_uv_edges[:,1,:], axis=1))
    
    return boundary_length