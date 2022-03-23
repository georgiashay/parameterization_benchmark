import numpy as np
import igl
import scipy

def face_proj(f):
    num_faces = f.shape[0]
    data = f.reshape((-1,))
    rows = np.arange(0, num_faces).repeat(3)
    cols = np.arange(0, num_faces).repeat(3)
    cols[1::3] += num_faces
    cols[2::3] += 2*num_faces
    return scipy.sparse.csr_matrix((data, (rows, cols)))

def get_jacobian(v, f, uv, ftc):
    exploded_v_idxs = f.reshape((-1))
    exploded_v = v[exploded_v_idxs]
    exploded_f = np.arange(0, f.size).reshape((-1, 3))
    
    exploded_uv_idxs = ftc.reshape((-1))
    exploded_uv = uv[exploded_uv_idxs]
    exploded_ftc = np.arange(0, ftc.size).reshape((-1, 3))

    G = igl.grad(exploded_v, exploded_f)
    f1, f2, f3 = igl.local_basis(exploded_v, exploded_f)

    f1 = f1.reshape((-1, 3))
    f2 = f2.reshape((-1, 3))
    f3 = f3.reshape((-1, 3))
    
    dx = face_proj(f1) @ G
    dy = face_proj(f2) @ G

    J = np.zeros((f.shape[0], 2, 2))

    J[:,0,0] = dx @ exploded_uv[:,0]
    J[:,0,1] = dy @ exploded_uv[:,0]
    J[:,1,0] = dx @ exploded_uv[:,1]
    J[:,1,1] = dy @ exploded_uv[:,1]
    
    return J