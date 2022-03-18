import os
import argparse

def produce_tutte_output(tutte_program, dataset_folder, output_folder):
    dataset_files = os.listdir(dataset_folder)

    for i, f in enumerate(dataset_files):
        print(i+1, "/", len(dataset_files), f, end="\r\n")
        fpath = os.path.join(dataset_folder, f)
        full_name, ext = os.path.splitext(f)
        if os.path.isfile(fpath) and ext == ".obj" and not f.endswith("_all.obj"):
            input_name = fpath
            output_name = os.path.join(output_folder, f)

            os.system(tutte_program + " " + input_name + " " + output_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run parameterization benchmark")
    parser.add_argument("--tutte-program", type=str, required=True, dest="tutte_program")
    parser.add_argument("--dataset-folder", type=str, required=True, dest="dataset_folder")
    parser.add_argument("--output-folder", type=str, required=True, dest="output_folder")

    args = parser.parse_args()
    dataset_folder = os.path.abspath(args.dataset_folder)
    output_folder = os.path.abspath(args.output_folder)
    tutte_program = os.path.abspath(args.tutte_program)
    
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    produce_tutte_output(tutte_program, dataset_folder, output_folder)    
