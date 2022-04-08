import numpy as np

def get_v_uv_map(f, ftc):
    v_to_uv = {}
    uv_to_v = {}
    for i, face in enumerate(f):
        for j, v_idx in enumerate(face):
            uv_idx = ftc[i][j]
            if v_idx in v_to_uv:
                v_to_uv[v_idx].append(uv_idx)
            else:
                v_to_uv[v_idx] = [uv_idx]
            
            uv_to_v[uv_idx] = v_idx
            
    max_uv_idx = max(uv_to_v.keys())
    uv_map_arr = np.full(max_uv_idx + 1, None)
    for uv_idx, v_idx in uv_to_v.items():
        uv_map_arr[uv_idx] = v_idx
        
    uv_map_arr = uv_map_arr.astype(int)

    return v_to_uv, uv_to_v, uv_map_arr