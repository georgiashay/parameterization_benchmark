import sys
import os

import argparse
import pandas as pd
import numpy as np
import scipy
import igl
from enum import Enum
import shutil
from multiprocessing import Pool

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

class TagChoice(Enum):
    NO_TAG = 0
    YES_TAG = 1
    ANY_TAG = 2

class TagChooser:
    def __init__(self, choice_no, choice_yes):
        if not isinstance(choice_no, str):
            raise ValueError("Tag choice for no must be str")
        if not isinstance(choice_yes, str):
            raise ValueError("Tag choice for yes must be str")
        self.choice_no = choice_no
        self.choice_yes = choice_yes
        
    def __call__(self, s):
        if s == self.choice_no:
            return TagChoice.NO_TAG
        elif s == self.choice_yes:
            return TagChoice.YES_TAG
        elif s == "any":
            return TagChoice.ANY_TAG
        raise ValueError("Tag must be " + self.choice_no + ", " + self.choice_yes + ", or any.")

def get_dataset_characteristics(dataset_folder):
    df = pd.DataFrame(columns=["Filename", "Object Number", "Mesh Name", "Chart Number", \
                               "Vertices", "Faces", "Euler Characteristic",
                               "Total Boundary Length", "Boundary Faces", "Interior Faces", \
                               "Closed"])
    dataset_files = os.listdir(dataset_folder)

    for i, fname in enumerate(dataset_files):
        print(i+1, "/", len(dataset_files), fname, end="\r\n")
        fpath = os.path.join(dataset_folder, fname)
        full_name, ext = os.path.splitext(fname)
        if os.path.isfile(fpath) and ext == ".obj" and not fname.endswith("_all.obj"):
            split_name = full_name.split("_")
            object_number = int(split_name[1])
            mesh_name = "_".join(split_name[2:-1])
            chart_number = int(split_name[-1])

            v, uv, n, f, ftc, fn = igl.read_obj(fpath)
            f = f.reshape((-1, 3))
            
            boundary_triangles = np.sum(np.any(igl.triangle_triangle_adjacency(f)[0].reshape((-1, 3)) == -1, axis=1))
            interior_triangles = len(f) - boundary_triangles
            is_closed = boundary_triangles == 0
            euler_characteristic = igl.euler_characteristic(f)
            boundary_length = len(igl.boundary_facets(f))
               
            row = [fname, object_number, mesh_name, chart_number, \
                   len(v), len(f), euler_characteristic,
                   boundary_length, boundary_triangles, interior_triangles, \
                   is_closed]
            
            row_series = pd.Series(row, index=df.columns)
            
            df = df.append(row_series, ignore_index=True)
        
    df.to_csv("mesh_characteristics.csv")
    
def get_uv_rows(fname, ofpath, fpath, df_columns, use_cut_dataset):
    v_io, uv_io, f_o, ftc_o, v_o, uv_o, mesh_areas_o, uv_areas_o = preprocess(ofpath)
    f_o_wnan, ftc_o_wnan, mesh_areas_o_wnan, uv_areas_o_wnan = f_o, ftc_o, mesh_areas_o, uv_areas_o

    try:                
        if not os.path.isfile(fpath):
            print("No parameterization provided for", fname)
            row = [fname, len(f_o), len(v_o), np.nan, np.nan, np.nan, np.nan, np.nan, \
                  np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

            row_series = pd.Series(row, index=list(df_columns))

            return row_series, None

        v_i, uv_i, f, ftc, v, uv, mesh_areas, uv_areas = preprocess(fpath)

        mesh_modified=False
        if (v_i.shape != v_io.shape) or (f.shape != f_o.shape) or \
        (not np.all(np.abs(v_i - v_io) <= 1e-8)) or (not np.all(f == f_o)):
            mesh_modified = True

        if np.any(np.isnan(uv_i)):
            print("Nan texture coordinates for", fname)
            row = [fname, len(f), len(v), np.nan, np.nan, np.nan, np.nan, np.nan, \
                  np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

            row_series = pd.Series(row, index=list(df_columns))

            return row_series, None

        J, nan_faces = get_jacobian(v, f, uv, ftc)
        J_o, nan_faces_o = get_jacobian(v_o, f_o, uv_o, ftc_o)
        
        J_o_wnan = J_o
        f_wnan, ftc_wnan, mesh_areas_wnan, uv_areas_wnan, J_wnan = f, ftc, mesh_areas, uv_areas, J
        
        f = np.delete(f, list(nan_faces), axis=0)
        f_o = np.delete(f_o, list(nan_faces), axis=0)
        ftc = np.delete(ftc, list(nan_faces), axis=0)
        ftc_o = np.delete(ftc_o, list(nan_faces), axis=0)
        mesh_areas = np.delete(mesh_areas, list(nan_faces), axis=1)
        uv_areas = np.delete(uv_areas, list(nan_faces), axis=1)
        mesh_areas_o = np.delete(mesh_areas_o, list(nan_faces), axis=1)
        uv_areas_o = np.delete(uv_areas_o, list(nan_faces), axis=1)
        J = np.delete(J, list(nan_faces), axis=0)
        J_o = np.delete(J_o, list(nan_faces), axis=0)
        
        area_distortions, area_errors, max_area_distortion, total_area_distortion = get_area_distortion(uv_areas, mesh_areas)

        singular_values, min_singular_value, max_singular_value = get_singular_values(J)
        singular_values_o, _, _ = get_singular_values(J_o)
                
        percent_flipped = get_flipped(J)
        percent_flipped = min(percent_flipped, 1 - percent_flipped)

#             overlap_area = get_overlap_area(ftc, uv, singular_values)

        angle_distortions, angle_errors, max_angle_distortion, total_angle_distortion = get_angle_distortion(singular_values, mesh_areas, v, f, uv, ftc)
        _, angle_errors_o, _, _ = get_angle_distortion(singular_values_o, mesh_areas_o, v_o, f_o, uv_o, ftc_o)

        resolution = get_resolution(v_i, f, uv_i, ftc)

        if not mesh_modified:
            _, _, artist_angle_match = get_artist_angle_match(angle_errors_o, angle_errors, mesh_areas)
            _, _, artist_area_match = get_artist_area_match(mesh_areas, uv_areas_o, uv_areas)
        else:
            artist_angle_match = np.nan
            artist_area_match = np.nan

        if not mesh_modified:
            hausdorff_distance = 0
        else:
            hausdorff_distance = igl.hausdorff(v_io, f_o_wnan, v_i, f_wnan)

        row = [fname, len(f_wnan), len(v), max_area_distortion, total_area_distortion, \
              min_singular_value, max_singular_value, percent_flipped, \
              #overlap_area, 
              max_angle_distortion, total_angle_distortion, \
              resolution, artist_area_match, artist_angle_match, hausdorff_distance, \
              mesh_modified]

        if not use_cut_dataset:
            # Use original faces for this metric due to dangling UV verts
            v_to_uv_o, uv_to_v_o, uv_to_v_arr_o = get_v_uv_map(f_o_wnan, ftc_o_wnan)
            v_to_uv, uv_to_v, uv_to_v_arr = get_v_uv_map(f_wnan, ftc_wnan)

            boundary_length_ratio, new_boundary_length_ratio = get_uv_boundary_length(uv, ftc_wnan, f_wnan, uv_to_v_arr)

            artist_cut_match_mesh = get_artist_cut_match_mesh(uv_to_v_arr_o, v_o, ftc_o_wnan, uv_to_v_arr, v, ftc_wnan)

            artist_cut_match_uv = get_artist_cut_match_uv(uv_o, ftc_o_wnan, uv, ftc_wnan)

            row += [new_boundary_length_ratio, artist_cut_match_mesh, artist_cut_match_uv]

        row_series = pd.Series(row, index=list(df_columns))
        
        new_tri_df = pd.DataFrame({"Filename": fname, "Triangle Number": range(singular_values.shape[0]), "Singular Value 1": singular_values[:, 0], "Singular Value 2": singular_values[:, 1], "Reason": ""})
            
        return row_series, new_tri_df

    except Exception as e:
        print("Exception for", fname)
#         raise e
        print(e)
        if use_cut_dataset:
            row = [fname, len(f_o_wnan), len(v_o), np.nan, np.nan, np.nan, np.nan, np.nan, \
                  np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        else:
            row = [fname, len(f_o_wnan), len(v_o), np.nan, np.nan, np.nan, np.nan, np.nan, \
                  np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, \
                  np.nan]

        row_series = pd.Series(row, index=list(df_columns))
        
        return row_series, None
    
def get_uv_characteristics(dataset_folder, measure_folder, use_cut_dataset, output_folder, tag_choices, processes):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    else:
        shutil.rmtree(output_folder)
        os.mkdir(output_folder)
    
    if use_cut_dataset:
        cut_choice_folder = "Cut"
        interesting_mesh_names = [("crumpled_developable", 0), ("castle_spiral_staircase", 10), \
                                  ("140mm_fan_cover", 0), ("bobcat_machine", 30), ("elephant_statue", 0), \
                                  ("gift", 5), ("moon", 1), ("gas_shock_absorber", 3), ("esme", 0), \
                                  ("fish", 1), ("human_tree", 40), ("horse_statue", 14), ("desert_rose", 28)]
    else:
        cut_choice_folder = "Uncut"
        interesting_mesh_names = [("crumpled_developable", 0), ("microphone", 19), ("future_motorcycle", 1), \
                                  ("140mm_fan_cover", 0), ("owl", 2), ("pile_of_sticks_and_logs", 0), \
                                  ("gift", 0), ("moon", 0), ("gas_shock_absorber", 1), ("fish", 0), \
                                  ("coronavirus", 0), ("heart_shaped_glasses", 0), ("swiss_cheese", 0), \
                                  ("drill", 0), ("lantern_chandelier", 13), ("vegetable_soup", 0)]
        
    dataset_subfolder = os.path.join(dataset_folder, "Artist_UVs", cut_choice_folder)
        
    tag_file = os.path.join(dataset_folder, "dataset_tags.csv")
    tag_df = pd.read_csv(tag_file)
    tag_df = tag_df.set_index(["Cut", "Filename"])
    tag_df = tag_df.sort_index()
        
    dataset_files = os.listdir(dataset_subfolder)
    
    interesting_meshes = []
    for mname, num in interesting_mesh_names:
        try:
            fname = next(filename for filename in dataset_files if filename.endswith(mname + "_" + str(num) + ".obj"))
        except StopIteration:
            continue
        interesting_meshes.append((fname, "Handpicked"))
    
    df_columns = ["Filename", "Faces", "Vertices", "Max Area Distortion", "Average Area Error", \
                  "Min Singular Value", "Max Singular Value", "Proportion Flipped Triangles", \
                  #"Bijectivity Violation Area", 
                  "Max Angle Distortion", "Average Angle Error", \
                  "Resolution", "Artist Area Match", "Artist Angle Match", "Hausdorff Distance", "Remeshed"]
    
    if not use_cut_dataset:
        df_columns += ["UV Cut Boundary Ratio", "Artist Mesh Cut Length Match", "Artist UV Cut Length Match"]
        
    df = pd.DataFrame(columns=df_columns)
    
    tri_df_columns = ["Filename", "Triangle Number", "Singular Value 1", "Singular Value 2", "Reason"]
    
    tri_df = pd.DataFrame(columns=tri_df_columns)

    
    with Pool(processes=processes) as pool:
        results = []
        for i, fname in enumerate(dataset_files):
            ofpath = os.path.join(dataset_subfolder, fname)
            fpath = os.path.join(measure_folder, fname)
            name, ext = os.path.splitext(fname)
            if ext == ".obj" and not fname.endswith("_all.obj"):
                tag_info = tag_df.loc[(cut_choice_folder, fname)]

                meets_tag = True
                for tag, choice in tag_choices.items():
                    if choice is TagChoice.YES_TAG and tag_info[tag] == False:
                        meets_tag = False
                        break
                    if choice is TagChoice.NO_TAG and tag_info[tag] == True:
                        meets_tag = False
                        break

                if not meets_tag:
                    continue

                results.append((fname, pool.apply_async(get_uv_rows, (fname, ofpath, fpath, tuple(df_columns), use_cut_dataset))))
        
        tri_df_sets = {}
        
        for i, (fname, res) in enumerate(results):
            print(i+1, "/", len(results), fname)
            df_row, tri_df_rows = res.get()
            df = df.append(df_row, ignore_index=True)
            if tri_df_rows is not None:
                tri_df_sets[fname] = tri_df_rows
                
    interesting_maxes = ["Artist Area Match", "Artist Angle Match", "Average Area Error", "Average Angle Error", \
                        "Proportion Flipped Triangles"]
    
    additional_interesting_meshes = [
        (df.iloc[np.argmax(df[max_key])]["Filename"], max_key)
        for max_key in interesting_maxes
    ]
    
    interesting_meshes += additional_interesting_meshes
    interesting_mesh_files = [fname for fname, reason in interesting_meshes]
                
    df.to_csv(os.path.join(output_folder, "distortion_characteristics.csv"))
    
    for fname, reason in interesting_meshes:
        if fname in tri_df_sets:
            new_rows = tri_df_sets[fname]
            new_rows["Reason"] = reason
            tri_df = tri_df.append(new_rows)
    
    tri_df.to_csv(os.path.join(output_folder, "triangle_singular_values.csv"))

    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run parameterization benchmark")
    parser.add_argument("-d", "--dataset", type=str, required=True, dest="dataset")
    parser.add_argument("-m", "--measure", type=str, required=True, dest="measure")
    parser.add_argument("-o", "--output", dest="output_folder", default="benchmark_output")
    parser.add_argument("-u", "--uncut", dest="uncut", action="store_const", const=True, default=False)
    parser.add_argument("--size", dest="size", default="any", type=TagChooser("large", "small"))
    parser.add_argument("--disk", dest="disk", default="any", type=TagChooser("no", "yes"))
    parser.add_argument("--closed", dest="closed", default="any", type=TagChooser("no", "yes"))
    parser.add_argument("--manifold", dest="manifold", default="any", type=TagChooser("no", "yes"))
    parser.add_argument("-p", "--processes", dest="processes", type=int, default=8)

    args = parser.parse_args()
    dataset_folder = os.path.abspath(args.dataset)
    measure_folder = os.path.abspath(args.measure)
    use_cut_dataset = not args.uncut
    output_folder = os.path.abspath(args.output_folder)
    processes = args.processes
    tag_choices = {
        "Disk": args.disk,
        "Closed": args.closed,
        "Manifold": args.manifold,
        "Small": args.size
    }
        
#     get_dataset_characteristics(dataset_folder)
    get_uv_characteristics(dataset_folder, measure_folder, use_cut_dataset, output_folder, tag_choices, processes)