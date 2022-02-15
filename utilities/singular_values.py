import numpy as np

def get_singular_values(J):
    singular_values = np.linalg.svd(J)[1]
    min_singular_value = np.min(singular_values)
    max_singular_value = np.max(singular_values)
    
    return singular_values, min_singular_value, max_singular_value