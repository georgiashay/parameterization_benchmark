import sys
import os

import argparse
import pandas as pd
import numpy as np
import scipy
import igl

from utilities.area_distortion import get_area_distortion
from utilities.preprocess import preprocess
from utilities.jacobian import get_jacobian
from utilities.singular_values import get_singular_values
from utilities.flipped import get_flipped
from utilities.angle_distortion import get_angle_distortion
from utilities.overlap_area import get_overlap_area
from utilities.resolution import get_resolution
from utilities.artist_match_area import get_artist_area_match
from utilities.artist_match_angle import get_artist_angle_match
from utilities.uv_boundary_ratio import get_uv_boundary_length
from utilities.artist_cut_match_mesh import get_artist_cut_match_mesh
from utilities.artist_cut_match_uv import get_artist_cut_match_uv
from utilities.v_uv_map import get_v_uv_map


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
    
def get_uv_characteristics(dataset_folder, measure_folder, use_cut_dataset, csv_name):
    if use_cut_dataset:
        dataset_subfolder = os.path.join(dataset_folder, "Cut")
    else:
        dataset_subfolder = os.path.join(dataset_folder, "Uncut")
        
    dataset_files = os.listdir(dataset_subfolder)
    
    columns = ["Filename", "Faces", "Vertices", "Max Area Distortion", "Total Area Distortion", \
               "Min Singular Value", "Max Singular Value", "Percentage Flipped Triangles", \
               #"Bijectivity Violation Area", 
               "Max Angle Distortion", "Total Angle Distortion", \
               "Resolution", "Artist Area Match", "Artist Angle Match"]
    
    if not use_cut_dataset:
        columns += ["UV Cut Boundary Ratio", "Artist Mesh Cut Length Match", "Artist UV Cut Length Match"]
        
    df = pd.DataFrame(columns=columns)
    
    tri_df = pd.DataFrame(columns=["Filename", "Triangle Number", "Singular Value 1", "Singular Value 2"])

    for i, fname in enumerate(dataset_files):
        print(i+1, "/", len(dataset_files), fname)
        ofpath = os.path.join(dataset_subfolder, fname)
        fpath = os.path.join(measure_folder, fname)
        name, ext = os.path.splitext(fname)
        if ext == ".obj" and not fname.endswith("_all.obj"):
            if not os.path.isfile(fpath):
                print("No parameterization provided for", fname)
                row = [fname, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, \
                      np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
            
                row_series = pd.Series(row, index=df.columns)

                df = df.append(row_series, ignore_index=True)
                continue
                
            v_io, uv_io, f_o, ftc_o, v_o, uv_o, mesh_areas_o, uv_areas_o = preprocess(ofpath)
            v_i, uv_i, f, ftc, v, uv, mesh_areas, uv_areas = preprocess(fpath)
            
            if not np.all(np.abs(v_i - v_io) <= 1e-8):
                print("Mesh modified for", fname)
                row = [fname, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, \
                      np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
            
                row_series = pd.Series(row, index=df.columns)

                df = df.append(row_series, ignore_index=True)
                continue
                
            if np.any(np.isnan(uv_i)):
                print("Nan texture coordinates for", fname)
                row = [fname, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, \
                      np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
            
                row_series = pd.Series(row, index=df.columns)

                df = df.append(row_series, ignore_index=True)
                continue
           
            area_distortions, area_errors, max_area_distortion, total_area_distortion = get_area_distortion(uv_areas, mesh_areas)
            
            J = get_jacobian(v, f, uv, ftc)
            J_o = get_jacobian(v_o, f_o, uv_o, ftc_o)

            singular_values, min_singular_value, max_singular_value = get_singular_values(J)
            singular_values_o, _, _ = get_singular_values(J_o)
            
            percent_flipped = get_flipped(J)
            percent_flipped = min(percent_flipped, 1 - percent_flipped)
            
#             overlap_area = get_overlap_area(ftc, uv, singular_values)
            
            angle_distortions, angle_errors, max_angle_distortion, total_angle_distortion = get_angle_distortion(singular_values, mesh_areas, v, f, uv, ftc)
            _, angle_errors_o, _, _ = get_angle_distortion(singular_values_o, mesh_areas_o, v_o, f_o, uv_o, ftc_o)
            
            resolution = get_resolution(v_i, f, uv_i, ftc)
            _, _, artist_angle_match = get_artist_angle_match(angle_errors_o, angle_errors, mesh_areas)
            _, _, artist_area_match = get_artist_area_match(mesh_areas, uv_areas_o, uv_areas)
                
            row = [fname, len(f), len(v), max_area_distortion, total_area_distortion, \
                  min_singular_value, max_singular_value, percent_flipped, \
                  #overlap_area, 
                  max_angle_distortion, total_angle_distortion, \
                  resolution, artist_area_match, artist_angle_match]
            
            if not use_cut_dataset:
                    
                v_to_uv_o, uv_to_v_o, uv_to_v_arr_o = get_v_uv_map(f_o, ftc_o)
                v_to_uv, uv_to_v, uv_to_v_arr = get_v_uv_map(f, ftc)
                
                boundary_length_ratio, new_boundary_length_ratio = get_uv_boundary_length(uv, ftc, f, uv_to_v_arr)

                artist_cut_match_mesh = get_artist_cut_match_mesh(uv_to_v_arr_o, v_o, ftc_o, uv_to_v_arr, v, ftc)
    
                artist_cut_match_uv = get_artist_cut_match_uv(uv_o, ftc_o, uv, ftc)
                
                row += [new_boundary_length_ratio, artist_cut_match_mesh, artist_cut_match_uv]
            
            row_series = pd.Series(row, index=df.columns)
            
            df = df.append(row_series, ignore_index=True)
            
            new_tri_df = pd.DataFrame({"Filename": fname, "Triangle Number": range(singular_values.shape[0]), "Singular Value 1": singular_values[:, 0], "Singular Value 2": singular_values[:, 1]})
            
            tri_df = tri_df.append(new_tri_df, ignore_index=True)
            
            
    df.to_csv(csv_name)
    tri_df.to_csv("triangle_singular_values.csv")

    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run parameterization benchmark")
    parser.add_argument("-d", "--dataset", type=str, required=True, dest="dataset")
    parser.add_argument("-m", "--measure", type=str, required=True, dest="measure")
    parser.add_argument("-u", "--uncut", dest="uncut", action="store_const", const=True, default=False)
    parser.add_argument("-o", "--output", dest="output", default="distortion_characteristics.csv")

    args = parser.parse_args()
    dataset_folder = os.path.abspath(args.dataset)
    measure_folder = os.path.abspath(args.measure)
    use_cut_dataset = not args.uncut
    csv_name = os.path.abspath(args.output)
    
#     get_dataset_characteristics(dataset_folder)
    get_uv_characteristics(dataset_folder, measure_folder, use_cut_dataset, csv_name)