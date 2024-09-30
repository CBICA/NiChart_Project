import argparse
import json
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", help="Provide the path to snakefile", required=True)
    parser.add_argument("--dset_name", help="Provide a name for your dataset", required=True)
    parser.add_argument("--input_rois", help="Provide input csv name with ROIs", required=True)
    parser.add_argument("--input_demog", help="Provide input csv name with demographic info", required=True)
    parser.add_argument("--dir_output", help="Provide output folder name", required=True)

    options = parser.parse_args()

    # Create out dir
    if not os.path.exists(options.dir_output):
        os.makedirs(options.dir_output)

    # Change path
    os.chdir(options.run_dir)

    # Run workflow
    print(f"Running: snakemake")

    # cmd = "snakemake -np"
    # cmd = cmd + " --config dset_name=" + options.dset_name
    # cmd = cmd + " input_rois=" + options.input_rois
    # cmd = cmd + " input_demog=" + options.input_demog
    # cmd = cmd + " dir_output=" + options.dir_output

    cmd = "snakemake "
    cmd = cmd + " --config dset_name=" + options.dset_name
    cmd = cmd + " input_rois=" + options.input_rois
    cmd = cmd + " input_demog=" + options.input_demog
    cmd = cmd + " dir_output=" + options.dir_output
    cmd = cmd + " --cores 1"


    print('Running cmd: ' + cmd)

    os.system(cmd)
