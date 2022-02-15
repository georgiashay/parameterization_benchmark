import numpy as np

def get_uv_coordinates(f, ftc, uv):
    uv_to_v = {}
    for i, face in enumerate(f):
        for j, v_idx in enumerate(face):
            uv_idx = ftc[i][j]
            uv_to_v[uv_idx] = v_idx

    uv_c = np.array([co for i, co in sorted(enumerate(uv), key=lambda i_co: uv_to_v[i_co[0]])])
    return uv_c