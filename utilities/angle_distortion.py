import numpy as np

def get_angle_distortion(singular_values, mesh_areas):
    angle_distortions = singular_values[:, 0]/singular_values[:, 1] + singular_values[:, 1]/singular_values[:, 0]
    max_angle_distortion = np.max(angle_distortions)

    finite_distortions = angle_distortions.copy()
    finite_distortions[np.where(finite_distortions > 1e32)] = 1e32
    total_angle_distortion = np.sum(mesh_areas * (finite_distortions - 2))
    
    return angle_distortions, max_angle_distortion, total_angle_distortion