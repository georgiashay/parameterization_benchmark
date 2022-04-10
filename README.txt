The benchmark is written as a command line tool. To use the benchmark, a folder of parameterized meshes must first be generated through some parameterization method. These meshes should contain the original meshes from the dataset, either Cut or Uncut, plus texture coordinates for the parameterization. These meshes should have the exact same names as the original meshes in the dataset. If desired, you can check the provided dataset_tags.csv file in the dataset and only run the parameterization method on meshes which have certain tags - i.e. large meshes, manifold meshes, etc.

First, create a conda environment that contains all the required python packages to run the benchmark by running the following commands:

    conda env create -f environment.yml
    conda activate pbench

The benchmark can then be run through the command line as follows:

    python3 benchmark.py [-h] -d DATASET -m MEASURE [-o OUTPUT_FOLDER] [-u] [--size SIZE] [--disk DISK] [--closed CLOSED] [--manifold MANIFOLD] [-p PROCESSES]

For the DATASET parameter, pass the path of the root folder of the dataset itself. For the MEASURE parameter, pass the path of the folder containing the parameterized meshes. Pass the -u flag to run the benchmark for uncut meshes. For the OUTPUT_FOLDER parameter, the user can specify the path to a folder (which will be created if it does not exist) where the results of the benchmark will be outputted. Finally, optionally specify the flags a mesh must have to be included in the benchmark results. For the SIZE parameter, you may pass 'small', 'large', or 'any', and for the DISK, CLOSED, and MANIFOLD parameters the user may pass 'yes', 'no', or 'any'. Any flag which is not passed defaults to 'any': i.e. meshes with any value for this flag will be includedd. To customize parallelization of the benchmark, you may also pass in the number of subprocesses to spawn to compute the benchmark results as the PROCESS parameter, which defaults to 8.

Once the benchmark has finished, there will be two files in the specified output folder: distortion_characteristics.csv, which contains per-mesh metrics and triangle_singular_values.csv, which contains the singular values of each triangle in the chosen "interesting meshes" for the benchmark.

To generate plots and a summary report of the benchmark information, run:

python3 generate_report.py [-h] -b INPUT_FOLDER [INPUT_FOLDER ...] -n NAME [NAME ...] [-o OUTPUT_FOLDER] [--force-no-comp]

The INPUT_FOLDER parameter should be the path to the folder that the benchmark just created with the CSV results. The NAME parameter should be the name of the parameterization method as it should appear on the plots in the report. Pass a destination folder for the report and plot images for the OUTPUT_FOLDER parameter. The benchmark will automatically compare the parameterization method to the artist's map. If you do not wish this to happen, pass the --force-no-comp switch.

If you would like to compare two parameterization methods using the benchmark, first run the benchmark.py script on each of the methods separately and output the results to different folders. Then, you may pass the two different benchmark result folders after the -b flag, and the two names of the methods after the -n flag. The two folders (names) are separated by a space and the flag does not need to be repeated. The report script will then generate scatterplots which compare the two methods on different metrics.