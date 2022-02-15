import numpy as np
import igl

def preprocess(fpath):
    """
    Load .obj file at fpath and preprocess to scale mesh and UV area to 1.0
    """
    v, uv, n, f, ftc, fn = igl.read_obj(fpath)
            
    f = f.reshape((-1, 3))
    ftc = ftc.reshape((-1, 3))

    mesh_areas = np.abs(igl.doublearea(v, f)/2.0).reshape((1, -1))
    uv_areas = np.abs(igl.doublearea(uv, ftc)/2.0).reshape((1, -1))

    total_mesh_area = np.sum(mesh_areas)
    total_uv_area = np.sum(uv_areas)

    v *= np.sqrt(1.0/total_mesh_area)
    uv *= np.sqrt(1.0/total_uv_area)
    mesh_areas *= 1.0/total_mesh_area
    uv_areas *= 1.0/total_uv_area
    
    return v, uv, f, ftc, mesh_areas, uv_areas