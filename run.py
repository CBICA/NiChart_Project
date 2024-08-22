import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", help="Select input folder", required=True)
    parser.add_argument("--cores", help="Select number of cores that the workflow will use to run", required=True)
    parser.add_argument("--no_conda", help="Run workflows without initializing a coda environment again", default = 0)

    options = parser.parse_args()

    if int(options.cores) < 1:
        print("Please select a valid number of cores(>=1)")
        exit(0)

    if options.no_conda == 0:
        os.system("conda install -n base -c conda-forge mamba")
        os.system("mamba create -c conda-forge -c bioconda -n NiChart_Workflows python=3.8")
        os.system("conda activate NiChart_Workflows")
        os.system("mamba install -c conda-forge -c bioconda snakemake")

    os.chdir('src/workflow/workflows/w_sMRI')
    os.system(f"snakemake --cores {options.cores} --config vTest={options.folder}")
