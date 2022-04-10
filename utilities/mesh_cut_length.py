import igl
import numpy as np

def get_mesh_cut_length(uv_to_v_arr, v, f, ftc): 
    uv_boundary_edges = igl.boundary_facets(ftc)
    v_boundary_edges = igl.boundary_facets(f)
    v_boundary_list = v_boundary_edges.tolist()
    
    on_original_boundary = [row.tolist() in v_boundary_list for row in uv_to_v_arr[uv_boundary_edges]]
    uv_new_boundary_edges = np.delete(uv_boundary_edges, np.where(on_original_boundary)[0], axis=0)

    v_edges = uv_to_v_arr[uv_new_boundary_edges]
    boundary_edges = v[v_edges]
    boundary_length = np.sum(np.linalg.norm(boundary_edges[:,0,:] - boundary_edges[:,1,:], axis=1))/2
 
    return boundary_length