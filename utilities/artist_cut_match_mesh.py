def get_artist_cut_match_mesh(v_artist, f_artist, v_measure, f_measure):
    boundary_edges_artist = v_artist[igl.boundary_facets(f_artist)]
    boundary_length_artist = np.sum(np.linalg.norm(boundary_edges_artist[:,0,:] - boundary_edges_artist[:,1,:], axis=1))
    
    boundary_edges_measure = v_measure[igl.boundary_facets(f_measure)]
    boundary_length_measure = np.sum(np.linalg.norm(boundary_edges_measure[:,0,:] - boundary_edges_measure[:,1,:], axis=1))
    
    return max(boundary_length_measure - boundary_length_artist, 0)