import os
import igl
import argparse
import subprocess
import glob
import shutil

def ceps(f_in, f_out, binary):
    try:
        subprocess.run([binary, f_in, "--outputMeshFilename", f_out], timeout=300)
    except subprocess.TimeoutExpired:
        print("Timed out for", f_in)
        
def produce_ceps_output(dataset_folder, output_folder, binary):    
    dataset_files = os.listdir(dataset_folder)

    for i, f in enumerate(dataset_files):
        print(i+1, "/", len(dataset_files), f, end="\r\n")
        fpath = os.path.join(dataset_folder, f)
        full_name, ext = os.path.splitext(f)
        if os.path.isfile(fpath) and ext == ".obj" and not f.endswith("_all.obj"):
            input_name = fpath
            output_name = os.path.join(output_folder, f)
            
            ceps(input_name, output_name, binary)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CEPS on dataset")
    parser.add_argument("-d", "--dataset-folder", type=str, required=True, dest="dataset_folder")
    parser.add_argument("-b", "--binary", type=str, required=True, dest="binary")
    parser.add_argument("-o", "--output-folder", type=str, required=True, dest="output_folder")

    args = parser.parse_args()
    dataset_folder = os.path.abspath(args.dataset_folder)
    output_folder = os.path.abspath(args.output_folder)
    binary = args.binary
    
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    produce_ceps_output(dataset_folder, output_folder, binary)    
