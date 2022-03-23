import numpy as np
EPSILON = 1e-8

def get_artist_angle_match(angle_errors_artist, angle_errors_measure, mesh_areas):
    angle_error_diff = angle_errors_measure - angle_errors_artist
    
    discard_indices = np.where(angle_error_diff < 0)
    angle_error_diff[discard_indices] = 0
    
    max_diff = np.max(angle_error_diff)
    total_diff = np.sum(mesh_areas * angle_error_diff)
    
    return angle_error_diff, max_diff, total_diff