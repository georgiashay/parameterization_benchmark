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
    nonzero_side_lengths = np.logical_and(np.logical_and(a != 0, b != 0), c != 0)
    
    A_div = np.divide(b2 + c2 - a2, 2*b*c, out=np.full(a.shape, np.nan), where=nonzero_side_lengths)
    A_div[np.where(A_div < -1)] = -1
    A_div[np.where(A_div > 1)] = 1
    A = np.arccos(A_div, out=np.full(a.shape, np.nan), where=nonzero_side_lengths)
    
    B_div = np.divide(a2 + c2 - b2, 2*a*c, out=np.full(a.shape, np.nan), where=nonzero_side_lengths)
    B_div[np.where(B_div < -1)] = -1
    B_div[np.where(B_div > 1)] = 1
    B = np.arccos(B_div, out=np.full(a.shape, np.nan), where=nonzero_side_lengths)
    
    C_div = np.divide(a2 + b2 - c2, 2*a*b, out=np.full(a.shape, np.nan), where=nonzero_side_lengths)
    C_div[np.where(C_div < -1)] = -1
    C_div[np.where(C_div > 1)] = 1
    C = np.arccos(C_div, out=np.full(a.shape, np.nan), where=nonzero_side_lengths)
    
    angles = np.concatenate((A.reshape((-1, 1)), B.reshape((-1, 1)), C.reshape((-1, 1))), axis=1)
    
    valid_col = nonzero_side_lengths.reshape((-1, 1))
    angles_valid = np.concatenate((valid_col, valid_col, valid_col), axis=1)
    return angles, angles_valid

def get_angle_distortion(singular_values, mesh_areas, v, f, uv, ftc):
    sing1 = singular_values[:, 0]
    sing2 = singular_values[:, 1]
        
    angle_distortions = np.divide(sing1, sing2, out=np.full(sing1.shape, np.inf), where=(sing2 != 0)) + \
                        np.divide(sing2, sing1, out=np.full(sing2.shape, np.inf), where=(sing1 != 0))
    
    max_angle_distortion = np.max(angle_distortions) - 2

    angles_mesh, angles_valid_mesh = get_angles(v, f)
    angles_uv, angles_valid_uv = get_angles(uv, ftc)
    
    angles_valid = np.logical_and(angles_valid_mesh, angles_valid_uv)
    angle_diffs = np.subtract(angles_mesh, angles_uv, out=np.full(angles_mesh.shape, 2*math.pi), where=angles_valid)
        
    angle_errors = np.sum(np.abs(angle_diffs), axis=1)
    average_angle_error = np.sum(mesh_areas * angle_errors)
                                
    return angle_distortions, angle_errors, max_angle_distortion, average_angle_error