import numpy as np
from shapely import geometry, ops

def get_overlap_area(ftc, uv, singular_values):
    magnification = singular_values[:, 0] * singular_values[:, 1]
    
    v_tri_p_indices = np.vstack((ftc[:, 0], ftc[:, 1], ftc[:, 2])).T
    
    tri_points = uv[v_tri_p_indices]
    
    triangles = []
    for tri in tri_points:
        point_list = [tuple(row) for row in tri]
        triangle = geometry.Polygon(point_list)
        triangles.append(triangle)

    overlap_area = 0
    for i, tri1 in enumerate(triangles):
        for j, tri2 in enumerate(triangles):
            if i != j:
                intersect_area = tri1.intersection(tri2).area
                overlap_area += intersect_area/magnification[i]
                
    return overlap_area