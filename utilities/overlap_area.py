import numpy as np
from shapely import geometry, ops

def get_overlap_area(f, uv_c):
    v_tri_p_indices = np.vstack((f[:, 0], f[:, 1], f[:, 2])).T
    
    tri_points = uv_c[v_tri_p_indices]
    
    triangles = []
    for tri in tri_points:
        point_list = [tuple(row) for row in tri]
        triangle = geometry.Polygon(point_list)
        triangles.append(triangle)

    uv_triangles = geometry.MultiPolygon(triangles)
    
    real_area = ops.unary_union(triangles).area
    overlap_area = 1 - real_area
    
    if overlap_area < 0:
        # Floating point issues
        overlap_area = 0
        
    return overlap_area