import sys
import os

import argparse
import pandas as pd
import numpy as np
# import pymesh
import scipy
import igl

from utilities.area_distortion import get_area_distortion
from utilities.preprocess import preprocess
from utilities.jacobian import get_jacobian
from utilities.uv_coordinates import get_uv_coordinates
from utilities.singular_values import get_singular_values
from utilities.flipped import get_flipped
from utilities.angle_distortion import get_angle_distortion
from utilities.overlap_area import get_overlap_area


def get_dataset_characteristics(dataset_folder):
    df = pd.DataFrame(columns=["Filename", "Object Number", "Mesh Name", "Chart Number", \
                               "Vertices", "Faces", "Euler Characteristic",
                               "Total Boundary Length", "Boundary Faces", "Interior Faces", \
                               "Closed"])
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

            v, uv, n, f, ftc, fn = igl.read_obj(fpath)
            
            boundary_triangles = np.sum(np.any(igl.triangle_triangle_adjacency(f)[0].reshape((-1, 3)) == -1, axis=1))
            interior_triangles = len(f) - boundary_triangles
            is_closed = boundary_triangles == 0
            euler_characteristic = igl.euler_characteristic(f)
            boundary_length = len(igl.boundary_facets(f))
               
            row = [f, object_number, mesh_name, chart_number, \
                   len(v), len(f), euler_characteristic,
                   boundary_length, boundary_triangles, interior_triangles, \
                   is_closed]
            
            row_series = pd.Series(row, index=df.columns)
            
            df = df.append(row_series, ignore_index=True)
        
    df.to_csv("mesh_characteristics.csv")
    
def get_uv_characteristics(dataset_folder, measure_folder):
    dataset_files = os.listdir(dataset_folder)
    
        
    df = pd.DataFrame(columns=["Filename", "Max Area Distortion", "Total Area Distortion", \
                               "Min Singular Value", "Max Singular Value", "Percentage Flipped Triangles",
                               "Bijectivity Violation Area", "Max Angle Distortion", "Total Angle Distortion"])
    
    tri_df = pd.DataFrame(columns=["Filename", "Triangle Number", "Singular Value 1", "Singular Value 2"])
    

    for i, fname in enumerate(dataset_files):
        print(i+1, "/", len(dataset_files), fname)
        fpath = os.path.join(dataset_folder, fname)
        name, ext = os.path.splitext(fname)
        if os.path.isfile(fpath) and ext == ".obj" and not fname.endswith("_all.obj"):
            v, uv, f, ftc, mesh_areas, uv_areas = preprocess(fpath)
            
            area_distortions, max_area_distortion, total_area_distortion = get_area_distortion(uv_areas, mesh_areas)
            
            uv_c = get_uv_coordinates(f, ftc, uv)
            
            J = get_jacobian(v, f, uv_c)
            
            singular_values, min_singular_value, max_singular_value = get_singular_values(J)
            
            percent_flipped = get_flipped(J)
            
            overlap_area = get_overlap_area(f, uv_c)
            
            angle_distortions, max_angle_distortion, total_angle_distortion = get_angle_distortion(singular_values, mesh_areas)
            
            row = [fname, max_area_distortion, total_area_distortion, \
                  min_singular_value, max_singular_value, percent_flipped, \
                  overlap_area, max_angle_distortion, total_angle_distortion]
            
            row_series = pd.Series(row, index=df.columns)
            
            df = df.append(row_series, ignore_index=True)
            
            new_tri_df = pd.DataFrame({"Filename": fname, "Triangle Number": range(singular_values.shape[0]), "Singular Value 1": singular_values[:, 0], "Singular Value 2": singular_values[:, 1]})
            
            tri_df = tri_df.append(new_tri_df, ignore_index=True)
            
            
    df.to_csv("distortion_characteristics.csv")
    tri_df.to_csv("triangle_singular_values.csv")

    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run parameterization benchmark")
    parser.add_argument("--dataset", type=str, required=True, dest="dataset")
    parser.add_argument("--measure", type=str, required=True, dest="measure")

    args = parser.parse_args()
    dataset_folder = os.path.abspath(args.dataset)
    measure_folder = os.path.abspath(args.measure)
    
    get_dataset_characteristics(dataset_folder, measure_folder)
#     get_uv_characteristics(dataset_folder)