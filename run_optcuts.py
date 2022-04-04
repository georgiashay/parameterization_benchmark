import os
import igl
import argparse
import subprocess
import glob
import shutil

def optcuts(f_in, f_out, binary, results_folder):
    name, ext = os.path.splitext(os.path.basename(f_in))

    try:
        subprocess.run([binary, "100", f_in, "0.999", "1", "0", "4.1", "1", "0", "pbench"], \
                       capture_output=True, timeout=60)
    except subprocess.TimeoutExpired:
        print("Timed out for", name)
        return
    
    
    folder_candidates = glob.glob(results_folder + "/" + name + "_*_pbench")
    
    if len(folder_candidates):
        output_folder = folder_candidates[0]
        output_file = os.path.join(output_folder, "finalResult_mesh.obj")
        if os.path.exists(output_file):
            subprocess.run(["cp", output_file, f_out])
        else:
            print("No output file for", name)
    else:
        print("Failed for", name)
        
def clear_optcuts_output(results_folder):
    if os.path.exists(results_folder):
        for root, dirs, files in os.walk(results_folder):
            for f in files:
                os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))
        
def produce_optcuts_output(dataset_folder, output_folder, binary, results_folder):
    clear_optcuts_output(results_folder)
    
    dataset_files = os.listdir(dataset_folder)

    for i, f in enumerate(dataset_files):
        print(i+1, "/", len(dataset_files), f, end="\r\n")
        fpath = os.path.join(dataset_folder, f)
        full_name, ext = os.path.splitext(f)
        if os.path.isfile(fpath) and ext == ".obj" and not f.endswith("_all.obj"):
            input_name = fpath
            output_name = os.path.join(output_folder, f)
            
            optcuts(input_name, output_name, binary, results_folder)
            
    subprocess.run(["rm", "-rf", results_folder])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Optcuts on dataset")
    parser.add_argument("-d", "--dataset-folder", type=str, required=True, dest="dataset_folder")
    parser.add_argument("-b", "--binary", type=str, required=True, dest="binary")
    parser.add_argument("-o", "--output-folder", type=str, required=True, dest="output_folder")

    args = parser.parse_args()
    dataset_folder = os.path.abspath(args.dataset_folder)
    output_folder = os.path.abspath(args.output_folder)
    binary = args.binary
    
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
        
    cwd = os.getcwd()
    results_folder = os.path.join(cwd, "output")
    
    produce_optcuts_output(dataset_folder, output_folder, binary, results_folder)    
