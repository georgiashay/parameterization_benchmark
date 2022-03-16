import igl
import numpy as np

def get_artist_cut_match_mesh(uv_to_v_arr_artist, v_artist, ftc_artist, uv_to_v_arr_measure, v_measure, ftc_measure):
    uv_edges_artist = igl.boundary_facets(ftc_artist)
    v_edges_artist = uv_to_v_arr_artist[uv_edges_artist]
    boundary_edges_artist = v_artist[v_edges_artist]
    boundary_length_artist = np.sum(np.linalg.norm(boundary_edges_artist[:,0,:] - boundary_edges_artist[:,1,:], axis=1))/2
    
    uv_edges_measure = igl.boundary_facets(ftc_measure)
    v_edges_measure = uv_to_v_arr_measure[uv_edges_measure]
    boundary_edges_measure = v_measure[v_edges_measure]
    boundary_length_measure = np.sum(np.linalg.norm(boundary_edges_measure[:,0,:] - boundary_edges_measure[:,1,:], axis=1))/2
        
    return max(boundary_length_measure - boundary_length_artist, 0)