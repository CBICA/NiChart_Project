import argparse
import json
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_input", help="Provide input folder", required=True)
    parser.add_argument("--dir_output", help="Provide output folder", required=True)
    parser.add_argument("--studies", type=int, help="Provide total studies", required=True)
    parser.add_argument("--version", type=str, default="test", help="Provide version")
    parser.add_argument("--cores", help="Select number of cores that the workflow will use to run", required=True)
    parser.add_argument("--conda", type=int, help="Run workflows without initializing a coda environment again", default=1)

    options = parser.parse_args()

    if int(options.cores) < 1:
        print("Please select a valid number of cores(>=1)")
        exit(0)

    if int(options.conda) == 1:
        os.system("pip install -r requirements.txt")
        os.system("conda install -n base -c conda-forge mamba")
        os.system("mamba init")
        os.system("mamba create -c conda-forge -c bioconda -n NiChart_Workflows python=3.8")
        os.system("conda activate NiChart_Workflows")
        os.system("mamba install -c conda-forge -c bioconda snakemake")

    config = {
        "dir_input": options.dir_input,
        "dir_output": options.dir_output
    }

    # generate info.json file for configuration
    if not os.path.exists("info.json"):
        os.system("touch info.json")

    config = {
        "version": options.version,
        "dir_input": options.dir_input,
        "dir_output": options.dir_output,
        "studies": options.studies,
        "list_ROIs_all": "MUSE/list_MUSE_all.csv",
        "list_ROIs_single": "MUSE/list_MUSE_single.csv",
        "list_ROIs_primary": "MUSE/list_MUSE_primary.csv",
        "list_derived_ROIs": "MUSE/list_MUSE_mapping_derived.csv",
        "corr_type": "normICV",
        "model_combat": "models/vISTAG1/COMBAT/combined_DLMUSE_raw_COMBATModel.pkl.gz",
        "model_SPARE-AD": "models/vISTAG1/SPARE/combined_DLMUSE_raw_COMBAT_SPARE-AD_Model.pkl.gz",
        "model_SPARE-Age": "models/vISTAG1/SPARE/combined_DLMUSE_raw_COMBAT_SPARE-Age_Model.pkl.gz",
        "model_SPARE-Diabetes": "models/vISTAG1/SPARE/combined_DLMUSE_raw_COMBAT_SPARE-Diabetes_Model.pkl.gz",
        "model_SPARE-Hyperlipidemia": "models/vISTAG1/SPARE/combined_DLMUSE_raw_COMBAT_SPARE-Hyperlipidemia_Model.pkl.gz",
        "model_SPARE-Hypertension": "models/vISTAG1/SPARE/combined_DLMUSE_raw_COMBAT_SPARE-Hypertension_Model.pkl.gz",
        "model_SPARE-Obesity": "models/vISTAG1/SPARE/combined_DLMUSE_raw_COMBAT_SPARE-Obesity_Model.pkl.gz",
        "model_SPARE-Smoking": "models/vISTAG1/SPARE/combined_DLMUSE_raw_COMBAT_SPARE-Smoking_Model.pkl.gz",
        "studies": [f"Study{i + 1}" for i in range(options.studies)],
        "SPARE_types": ["AD", "Age", "Diabetes", "Hyperlipidemia", "Hypertension", "Obesity", "Smoking"],
        "seg_types": ["DLMUSE"]
    }

    json_config = json.dumps(config, indent=4)
    with open("info.json", "w") as outfile:
        outfile.write(json_config)
    os.system("mv info.json src/workflow/workflows/")

    os.chdir('src/workflow/workflows/w_sMRI')
    os.system(f"snakemake --cores {options.cores}")
