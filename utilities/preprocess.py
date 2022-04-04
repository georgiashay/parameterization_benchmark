import numpy as np
import igl
import tempfile

def preprocess(fpath):
    """
    Load .obj file at fpath and preprocess to scale mesh and UV area to 1.0
    """
    try:
        v_i, uv_i, n, f, ftc, fn = igl.read_obj(fpath)
    except ValueError:
        new_f = ""
        with open(fpath) as f:
            for line in f:
                if line.startswith("f "):
                    verts = line.strip().split(" ")[1:]
                    if len(verts) == 3:
                        new_f += line
                    else:
                        for i in range(len(verts) - 2):
                            new_face = "f " + verts[0] + " " + verts[i+1] + " " + verts[i+2] + "\n"
                            new_f += new_face            
                else:
                    new_f += line
        tmp = tempfile.NamedTemporaryFile(mode="w+")
        tmp.write(new_f)
        v_i, uv_i, n, f, ftc, fn = igl.read_obj(tmp.name)
        tmp.close()
                
    f = f.reshape((-1, 3))
    ftc = ftc.reshape((-1, 3))

    mesh_areas = np.abs(igl.doublearea(v_i, f)/2.0).reshape((1, -1))
    uv_areas = np.abs(igl.doublearea(uv_i, ftc)/2.0).reshape((1, -1))

    total_mesh_area = np.sum(mesh_areas)
    total_uv_area = np.sum(uv_areas)

    v = v_i * np.sqrt(1.0/total_mesh_area)
    uv = uv_i * np.sqrt(1.0/total_uv_area)
    mesh_areas *= 1.0/total_mesh_area
    uv_areas *= 1.0/total_uv_area
    
    return v_i, uv_i, f, ftc, v, uv, mesh_areas, uv_areas