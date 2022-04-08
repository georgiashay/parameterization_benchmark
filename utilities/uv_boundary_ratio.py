import igl
import numpy as np

def get_uv_boundary_length(uv, ftc, f, uv_to_v_arr):  
    uv_boundary_edges = igl.boundary_facets(ftc)
    v_boundary_edges = igl.boundary_facets(f)
    v_boundary_list = v_boundary_edges.tolist()
    
    on_original_boundary = [row.tolist() in v_boundary_list for row in uv_to_v_arr[uv_boundary_edges]]
    uv_new_boundary_edges = np.delete(uv_boundary_edges, np.where(on_original_boundary)[0], axis=0)
    
    uv_boundary_coords = uv[uv_boundary_edges]
    boundary_length = np.sum(np.linalg.norm(uv_boundary_coords[:,0,:] - uv_boundary_coords[:,1,:], axis=1))
    
    uv_new_boundary_coords = uv[uv_new_boundary_edges]
    new_boundary_length = np.sum(np.linalg.norm(uv_new_boundary_coords[:,0,:] - uv_new_boundary_coords[:,1,:], axis=1))
    
    return boundary_length, new_boundary_length