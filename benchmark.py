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
from utilities.symmetric_dirichlet_energy import get_symmetric_dirichlet_energy
from utilities.resolution import get_resolution
from utilities.artist_correlation import get_artist_correlation
from utilities.mesh_cut_length import get_mesh_cut_length
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
    
def get_uv_rows(fname, ofpath, fpath, df_columns, use_cut_dataset):
    v_io, uv_io, f_o, ftc_o, v_o, uv_o, mesh_areas_o, uv_areas_o = preprocess(ofpath)
    f_o_wnan, ftc_o_wnan, mesh_areas_o_wnan, uv_areas_o_wnan = f_o, ftc_o, mesh_areas_o, uv_areas_o

    try:                
        if not os.path.isfile(fpath):
            print("No parameterization provided for", fname)
            row = [fname, len(f_o), len(v_o)] + [np.nan] * (len(df_columns) - 3)

            df_row = pd.DataFrame({ col: [row[i]] for i, col in enumerate(df_columns)})
            
            return df_row, None

        v_i, uv_i, f, ftc, v, uv, mesh_areas, uv_areas = preprocess(fpath)

        mesh_modified=False
        if (v_i.shape != v_io.shape) or (f.shape != f_o.shape) or \
        (not np.all(np.abs(v_i - v_io) <= 1e-5)) or (not np.all(f == f_o)):
            mesh_modified = True

        if np.any(np.isnan(uv_i)):
            print("Nan texture coordinates for", fname)
            row = [fname, len(f_o), len(v_o)] + [np.nan] * (len(df_columns) - 3)

            df_row = pd.DataFrame({ col: [row[i]] for i, col in enumerate(df_columns)})

            return df_row, None

        J, nan_faces = get_jacobian(v, f, uv, ftc)
        J_o, nan_faces_o = get_jacobian(v_o, f_o, uv_o, ftc_o)
        
        J_o_wnan = J_o
        f_wnan, ftc_wnan, mesh_areas_wnan, uv_areas_wnan, J_wnan = f, ftc, mesh_areas, uv_areas, J
        
        if mesh_modified:
            f = np.delete(f, list(nan_faces), axis=0)
            f_o = np.delete(f_o, list(nan_faces_o), axis=0)
            ftc = np.delete(ftc, list(nan_faces), axis=0)
            ftc_o = np.delete(ftc_o, list(nan_faces_o), axis=0)
            mesh_areas = np.delete(mesh_areas, list(nan_faces), axis=1)
            uv_areas = np.delete(uv_areas, list(nan_faces), axis=1)
            mesh_areas_o = np.delete(mesh_areas_o, list(nan_faces_o), axis=1)
            uv_areas_o = np.delete(uv_areas_o, list(nan_faces_o), axis=1)
            J = np.delete(J, list(nan_faces), axis=0)
            J_o = np.delete(J_o, list(nan_faces_o), axis=0)
        else:
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

        symmetric_dirichlet_energy = get_symmetric_dirichlet_energy(singular_values, mesh_areas)
        
        nan_faces_insert_locs = [face - i for i, face in enumerate(sorted(nan_faces))]
        singular_values_wnan = np.insert(singular_values, nan_faces_insert_locs, values=np.nan, axis=0)
                
        percent_flipped = get_flipped(J)
        percent_flipped = min(percent_flipped, 1 - percent_flipped)

        angle_distortions, angle_errors, max_angle_distortion, total_angle_distortion = get_angle_distortion(singular_values, mesh_areas, v, f, uv, ftc)
        _, angle_errors_o, _, _ = get_angle_distortion(singular_values_o, mesh_areas_o, v_o, f_o, uv_o, ftc_o)

        resolution = get_resolution(v_i, f, uv_i, ftc)

        if not mesh_modified:
            artist_correlation = get_artist_correlation(singular_values_o, singular_values, mesh_areas)
        else:
            artist_correlation = np.nan

        row = [fname, len(f_wnan), len(v), max_area_distortion, total_area_distortion, \
              min_singular_value, max_singular_value, percent_flipped, \
              max_angle_distortion, total_angle_distortion, \
              resolution, artist_correlation, \
              mesh_modified, \
              symmetric_dirichlet_energy]

        if not use_cut_dataset:
            # Use original faces for this metric due to dangling UV verts
            v_to_uv_o, uv_to_v_o, uv_to_v_arr_o = get_v_uv_map(f_o_wnan, ftc_o_wnan)
            v_to_uv, uv_to_v, uv_to_v_arr = get_v_uv_map(f_wnan, ftc_wnan)

            mesh_cut_length = get_mesh_cut_length(uv_to_v_arr, v, f_wnan, ftc_wnan)

            artist_cut_match_mesh = get_artist_cut_match_mesh(uv_to_v_arr_o, v_o, ftc_o_wnan, uv_to_v_arr, v, ftc_wnan)

            row += [mesh_cut_length, artist_cut_match_mesh]

        df_row = pd.DataFrame({ col: [row[i]] for i, col in enumerate(df_columns)})
        
        new_tri_df = pd.DataFrame({"Filename": fname, "Triangle Number": range(singular_values_wnan.shape[0]), "Singular Value 1": singular_values_wnan[:, 0], "Singular Value 2": singular_values_wnan[:, 1], "Reason": ""})
            
        return df_row, new_tri_df

    except Exception as e:
        if str(e) != "No UV area" and str(e) != "No faces in mesh":
            raise e
        print("Exception for", fname)
        print(e)
        row = [fname, len(f_o_wnan), len(v_o)] + [np.nan] * (len(df_columns) - 3)

        df_row = pd.DataFrame({ col: [row[i]] for i, col in enumerate(df_columns)})
        
        return df_row, None
    
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
    
    df_columns = ["Filename", "Faces", "Vertices", "Max Area Distortion", "Average Area Discrepancy", \
                  "Min Singular Value", "Max Singular Value", "Proportion Flipped Triangles", \
                  "Max Angle Distortion", "Average Angle Discrepancy", \
                  "Resolution", "Artist Correlation", "Remeshed", \
                  "Symmetric Dirichlet Energy"]
    
    if not use_cut_dataset:
        df_columns += ["Mesh Cut Length", "Artist Mesh Cut Length Match"]
        
    df = pd.DataFrame(columns=df_columns)
    
    tri_df_columns = ["Filename", "Triangle Number", "Singular Value 1", "Singular Value 2", "Reason"]
    
    tri_df = pd.DataFrame(columns=tri_df_columns)

    
    with Pool(processes=processes) as pool:
        results = []
        for i, fname in enumerate(dataset_files):
            ofpath = os.path.join(dataset_subfolder, fname)
            fpath = os.path.join(measure_folder, fname)
            name, ext = os.path.splitext(fname)
            if ext == ".obj":
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
            df = pd.concat([df, df_row], ignore_index=True)
            if tri_df_rows is not None:
                tri_df_sets[fname] = tri_df_rows
                
    interesting_maxes = ["Artist Correlation", "Average Area Discrepancy", "Average Angle Discrepancy", \
                        "Proportion Flipped Triangles"]
    
    additional_interesting_meshes = [
        (df.iloc[np.argmax(df[max_key])]["Filename"], max_key)
        for max_key in interesting_maxes
    ]
    
    interesting_meshes += additional_interesting_meshes
    interesting_mesh_files = [fname for fname, reason in interesting_meshes]
        
    filename_sort_keys = df["Filename"].apply(lambda s: (int(s.split("_")[1]), int(s.split("_")[-1].split(".")[0])))
    sorted_index = filename_sort_keys.sort_values().index
    df = df.loc[sorted_index]
    
    df.to_csv(os.path.join(output_folder, "distortion_characteristics.csv"), index=False)
    
    for fname, reason in interesting_meshes:
        if fname in tri_df_sets:
            new_rows = tri_df_sets[fname]
            new_rows["Reason"] = reason
            tri_df = pd.concat([tri_df, new_rows])
    
    tri_df.to_csv(os.path.join(output_folder, "triangle_singular_values.csv"), index=False)

    
    
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
     
    artist_subfolder = os.path.join(output_folder, "artist")
    cut_choice = "Cut" if use_cut_dataset else "Uncut"
    artist_param_folder = os.path.join(dataset_folder, "Artist_UVs", cut_choice)
    
    get_uv_characteristics(dataset_folder, measure_folder, use_cut_dataset, output_folder, tag_choices, processes)
    get_uv_characteristics(dataset_folder, artist_param_folder, use_cut_dataset, artist_subfolder, tag_choices, processes)