import igl
import os
import argparse

def tutte(f_in, f_out):
    v, uv, n, f, ftc, fn = igl.read_obj(f_in)
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
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Tutte on file")
    parser.add_argument("input_file", type=str)
    parser.add_argument("output_file", type=str)

    args = parser.parse_args()
    
    input_file = os.path.abspath(args.input_file)
    output_file = os.path.abspath(args.output_file)
    
    tutte(input_file, output_file)