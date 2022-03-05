import numpy as np
EPSILON = 1e-8

def get_artist_angle_match(singular_values_artist, singular_values_measure):
    d_you = singular_values_measure[:, 0]/singular_values_measure[:, 1] + \
            singular_values_measure[:, 1]/singular_values_measure[:, 0]
    d_artist = singular_values_artist[:, 0]/singular_values_artist[:, 1] + \
                singular_values_artist[:, 1]/singular_values_artist[:, 0]
    
    angle_distortion_diff = d_you - d_artist
    
    discard_indices = np.where(np.logical_and(angle_distortion_diff >= 0, np.all(singular_values_artist > EPSILON, axis=1)))
    angle_distortion_diff[discard_indices] = 0
    
    max_diff = np.max(angle_distortion_diff)
    total_diff = np.sum(angle_distortion_diff)
    
    return angle_distortion_diff, max_diff, total_diff