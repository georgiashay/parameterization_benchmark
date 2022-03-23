import numpy as np
import math

def get_angles(v, f):
    side_lengths = np.linalg.norm(v[f[:, (0,1,2)]] - v[f[:, (1,2,0)]], axis=2)
    a = side_lengths[:, 0]
    a2 = np.square(a)
    b = side_lengths[:, 1]
    b2 = np.square(b)
    c = side_lengths[:, 2]
    c2 = np.square(c)
    A = np.arccos((b2 + c2 - a2)/(2*b*c))
    B = np.arccos((a2 + c2 - b2)/(2*a*c))
    C = np.arccos((a2 + b2 - c2)/(2*a*b))
    angles = np.concatenate((A.reshape((-1, 1)), B.reshape((-1, 1)), C.reshape((-1, 1))), axis=1)
    return angles

def get_angle_distortion(singular_values, mesh_areas, v, f, uv, ftc):
    angle_distortions = singular_values[:, 0]/singular_values[:, 1] + singular_values[:, 1]/singular_values[:, 0]
    max_angle_distortion = np.max(angle_distortions) - 2

    angles_mesh = get_angles(v, f)
    angles_uv = get_angles(uv, ftc)
        
    angle_errors = np.sum(np.abs(angles_mesh - angles_uv), axis=1)
    average_angle_error = np.sum(mesh_areas * angle_errors)
                                
    return angle_distortions, angle_errors, max_angle_distortion, average_angle_error