import os
import igl
import argparse

def tutte(f_in, f_out):
    v, uv, n, f, ftc, fn = igl.read_obj(f_in)
    f = f.reshape((-1, 3))
    b = igl.boundary_loop(f)
    b_uv = igl.map_vertices_to_circle(v, b)
    v_uv = igl.harmonic_weights(v, f, b, b_uv, 1)
    
    with open(f_out, "w") as objfile:
        for row in v:
            objfile.write("v\t" + " ".join([str(c) for c in row]) + "\n")
        for row in v_uv:
            objfile.write("vt\t" + " ".join([str(c) for c in row]) + "\n")
        for row in f:
            objfile.write("f\t" + " ".join([str(i+1) + "/" + str(i+1) for i in row]) + "\n")

def produce_tutte_output(dataset_folder, output_folder):
    dataset_files = os.listdir(dataset_folder)

    for i, f in enumerate(dataset_files):
        print(i+1, "/", len(dataset_files), f, end="\r\n")
        fpath = os.path.join(dataset_folder, f)
        full_name, ext = os.path.splitext(f)
        if os.path.isfile(fpath) and ext == ".obj" and not f.endswith("_all.obj"):
            input_name = fpath
            output_name = os.path.join(output_folder, f)
            
            try:
                tutte(input_name, output_name)
            except ValueError as e:
                print(e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run parameterization benchmark")
    parser.add_argument("-d", "--dataset-folder", type=str, required=True, dest="dataset_folder")
    parser.add_argument("-o", "--output-folder", type=str, required=True, dest="output_folder")

    args = parser.parse_args()
    dataset_folder = os.path.abspath(args.dataset_folder)
    output_folder = os.path.abspath(args.output_folder)
    
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    produce_tutte_output(dataset_folder, output_folder)    
