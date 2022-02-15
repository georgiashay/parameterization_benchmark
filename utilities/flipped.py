import numpy as np

def get_flipped(J):
    dets = np.linalg.det(J)
    flipped = dets < 0
    percent_flipped = np.sum(flipped)/flipped.shape[0]
    
    return percent_flipped