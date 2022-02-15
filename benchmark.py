import sys
import os

import argparse
import pandas as pd
import numpy as np
import pymesh
import scipy
import igl

def tri_area(co1, co2, co3):
    return np.linalg.norm(np.cross((co2 - co1),( co3 - co1 )))/ 2.0

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
        return v
    return v/norm

def local_basis(vertices, faces):
    B1 = np.zeros((faces.shape[0], 3))
    B2 = np.zeros((faces.shape[0], 3))
    B3 = np.zeros((faces.shape[0], 3))
    
    for i in range(faces.shape[0]):
        v1 = normalize(vertices[faces[i][1]] - vertices[faces[i][0]])
        t = vertices[faces[i][2]] - vertices[faces[i][0]]
        v3 = normalize(np.cross(v1, t))
        v2 = normalize(np.cross(v1, v3))
    
        B1[i] = v1
        B2[i] = -v2
        B3[i] = v3
        
    return B1, B2, B3

def face_proj(f):
    num_faces = f.shape[0]
    data = f.reshape((-1,))
    rows = np.arange(0, num_faces).repeat(3)
    cols = np.arange(0, num_faces).repeat(3)
    cols[1::3] += num_faces
    cols[2::3] += 2*num_faces
    return scipy.sparse.csr_matrix((data, (rows, cols)))

def get_dataset_characteristics():
    df = pd.DataFrame(columns=["Filename", "Object Number", "Mesh Name", "Chart Number", \
                               "Vertices", "Faces", "Euler Characteristic", "Genus", \
                               "Total Boundary Length", "Boundary Faces", "Interior Faces", \
                               "Edge Manifold", "Vertex Manifold", "Closed", "Connectivity Valid"])
    dataset_files = os.listdir(dataset_folder)

    for i, f in enumerate(dataset_files):
        print(i+1, "/", len(dataset_files), f, end="\r\n")
        fpath = os.path.join(dataset_folder, f)
        full_name, ext = os.path.splitext(f)
        if os.path.isfile(fpath) and ext == ".obj" and not f.endswith("_all.obj"):
            split_name = full_name.split("_")
            object_number = int(split_name[1])
            mesh_name = "_".join(split_name[2:-1])
            chart_number = int(split_name[-1])

            mesh = pymesh.load_mesh(fpath)
            cut_mesh = pymesh.cut_mesh(mesh)
            
            connectivity_valid = np.all(cut_mesh.faces == mesh.faces)
            is_manifold = mesh.is_edge_manifold() and mesh.is_vertex_manifold()
            
            boundary_triangles = np.sum(np.any(igl.triangle_triangle_adjacency(mesh.faces)[0].reshape((-1, 3)) == -1, axis=1))
            interior_triangles = len(mesh.faces) - boundary_triangles
               
            row = [f, object_number, mesh_name, chart_number, \
                   len(mesh.vertices), len(mesh.faces), mesh.euler_characteristic, mesh.genus, \
                   mesh.num_boundary_edges, boundary_triangles, interior_triangles, \
                   mesh.is_edge_manifold(), mesh.is_vertex_manifold(), mesh.is_closed(), connectivity_valid]
            
            row_series = pd.Series(row, index=df.columns)
            
            df = df.append(row_series, ignore_index=True)
        
    df.to_csv("mesh_characteristics.csv")
    
    df = pd.DataFrame(columns=["Filename", "Max Area Distortion", "Total Area Distortion", \
                               "Min Singular Value", "Max Singular Value", "Percentage Flipped Triangles",
                               "Max Angle Distortion", "Total Angle Distortion"])
    
    tri_df = pd.DataFrame(columns=["Filename", "Triangle Number", "Singular Value 1", "Singular Value 2"])
    

    for i, fname in enumerate(dataset_files):
        print(i+1, "/", len(dataset_files), fname)
        fpath = os.path.join(dataset_folder, fname)
        name, ext = os.path.splitext(fname)
        if os.path.isfile(fpath) and ext == ".obj" and not fname.endswith("_all.obj"):
            mesh = pymesh.load_mesh(fpath)
            
            is_manifold = mesh.is_edge_manifold() and mesh.is_vertex_manifold()
            cut_mesh = pymesh.cut_mesh(mesh)
            connectivity_valid = np.all(cut_mesh.faces == mesh.faces)
            
            if not is_manifold or not connectivity_valid:
                continue
             
            v, uv, n, f, ftc, fn = igl.read_obj(fpath)
            
            f = f.reshape((-1, 3))
            ftc = ftc.reshape((-1, 3))
            
            mesh_areas = np.abs(igl.doublearea(v, f)/2.0).reshape((1, -1))
            uv_areas = np.abs(igl.doublearea(uv, ftc)/2.0).reshape((1, -1))
            
            total_mesh_area = np.sum(mesh_areas)
            total_uv_area = np.sum(uv_areas)
            
            v *= np.sqrt(1.0/total_mesh_area)
            uv *= np.sqrt(1.0/total_uv_area)
            mesh_areas *= 1.0/total_mesh_area
            uv_areas *= 1.0/total_uv_area
            
            area_distortions = np.abs(uv_areas - mesh_areas)
            max_area_distortion = np.max(area_distortions)
            total_area_distortion = np.sum(area_distortions)
            
            uv_to_v = {}
            for i, face in enumerate(f):
                for j, v_idx in enumerate(face):
                    uv_idx = ftc[i][j]
                    uv_to_v[uv_idx] = v_idx
                    
            uv_c = np.array([co for i, co in sorted(enumerate(uv), key=lambda i_co: uv_to_v[i_co[0]])])
            
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
            
            singular_values = np.linalg.svd(J)[1]
            min_singular_value = np.min(singular_values)
            max_singular_value = np.max(singular_values)
            
            dets = np.linalg.det(J)
            flipped = dets < 0
            percent_flipped = np.sum(flipped)/flipped.shape[0]
            
            angle_distortions = singular_values[:, 0]/singular_values[:, 1] + singular_values[:, 1]/singular_values[:, 0]
            max_angle_distortion = np.max(angle_distortions)
            
            finite_distortions = angle_distortions.copy()
            finite_distortions[np.where(finite_distortions > 1e32)] = 1e32
            total_angle_distortion = np.sum(mesh_areas * (finite_distortions - 2))
            
            row = [fname, max_area_distortion, total_area_distortion, \
                  min_singular_value, max_singular_value, percent_flipped, \
                  max_angle_distortion, total_angle_distortion]
            
            row_series = pd.Series(row, index=df.columns)
            
            df = df.append(row_series, ignore_index=True)
            
            new_tri_df = pd.DataFrame({"Filename": fname, "Triangle Number": range(singular_values.shape[0]), "Singular Value 1": singular_values[:, 0], "Singular Value 2": singular_values[:, 1]})
            
            tri_df = tri_df.append(new_tri_df, ignore_index=True)
            
            
    df.to_csv("distortion_characteristics.csv")
    tri_df.to_csv("triangle_singular_values.csv")

    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run parameterization benchmark")
    parser.add_argument("--dataset", type=str, required=True, dest="dataset")

    args = parser.parse_args()
    dataset_folder = os.path.abspath(args.dataset)
    
    get_dataset_characteristics()