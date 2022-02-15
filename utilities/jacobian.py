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

def get_jacobian(v, f, uv_c):
    G = igl.grad(v, f)
    f1, f2, f3 = igl.local_basis(v, f)

    f1 = f1.reshape((-1, 3))
    f2 = f2.reshape((-1, 3))
    f3 = f3.reshape((-1, 3))

    dx = face_proj(f1) @ G
    dy = face_proj(f2) @ G

    J = np.zeros((f.shape[0], 2, 2))

    J[:,0,0] = dx @ uv_c[:,0]
    J[:,0,1] = dy @ uv_c[:,0]
    J[:,1,0] = dx @ uv_c[:,1]
    J[:,1,1] = dy @ uv_c[:,1]
    
    return J