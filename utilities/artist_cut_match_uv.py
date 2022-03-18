import igl
import numpy as np

def get_artist_cut_match_uv(uv_artist, ftc_artist, uv_measure, ftc_measure):
    boundary_edges_artist = uv_artist[igl.boundary_facets(ftc_artist)]
    boundary_length_artist = np.sum(np.linalg.norm(boundary_edges_artist[:,0,:] - boundary_edges_artist[:,1,:], axis=1))/2
    
    boundary_edges_measure = uv_measure[igl.boundary_facets(ftc_measure)]
    boundary_length_measure = np.sum(np.linalg.norm(boundary_edges_measure[:,0,:] - boundary_edges_measure[:,1,:], axis=1))/2
    
    x = uv_artist[igl.oriented_facets(ftc_artist)]
    
    return max(boundary_length_measure - boundary_length_artist, 0)