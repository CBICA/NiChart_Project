# Import packages
import argparse
import os
from typing import Any

import utils_wsmri as utilw

dict_config = {
    "dset_name": "",
    "input_rois": "",
    "input_demog": "",
    "dir_output": "",
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
    "SPARE_types": ["AD", "Age"],   ## FIXME : short version during tests
    #"SPARE_types": ["AD", "Age", "Diabetes", "Hyperlipidemia", "Hypertension", "Obesity", "Smoking"],
    "seg_types": ["DLMUSE"],
}

def run_workflow(root_dir: Any, dict_config: Any) -> None:

    def step_combat():
        # Rename MUSE roi indices to roi codes
        utilw.rename_df_columns(input_rois, list_rois, "Index", "Code", f_raw)

        # Merge covars to ROIs
        key_var = "MRID"
        utilw.merge_dataframes(input_demog, f_raw, key_var, f_comb)

        # Select variables for harmonization
        dict_var = "Code"
        covars = "MRID,Age,Sex,SITE,MUSE_702"
        f_sel = os.path.join(out_dir, f"{dset_name}_sel.csv")
        utilw.select_vars(f_comb, dict_csv, dict_var, covars, f_sel)

        # Check that sample has age range consistent with the model
        var_name = "Age"
        min_val = 50
        max_val = 95
        f_filt = os.path.join(out_dir, f"{dset_name}_filt.csv")
        utilw.filter_num_var(f_sel, var_name, min_val, max_val, f_filt)

        # Apply combat
        mdl = os.path.join("src", "workflow", model_combat)
        f_combat1 = os.path.join(out_dir, f"{dset_name}_COMBAT_single.csv")
        utilw.apply_combat(f_filt, mdl, "MRID", "_HARM", f_combat1)

        # Calculate derived ROIs from harmonized data
        in_dict = os.path.join("src", "workflow", derived_rois)
        key_var = "MRID"
        # roi_prefix = "MUSE_"
        utilw.combine_rois(f_combat1, in_dict, f_combat_wderived)

        # Merge covars to ROIs
        key_var = "MRID"
        utilw.merge_dataframes(input_demog, f_combat_wderived, key_var, f_combat_wcovar)

    def step_spare():
        # Select variables for spare
        dict_var = "Code"
        covars = "MRID,Age,Sex,DLICV"
        utilw.select_vars(f_combat_wcovar, dict_csv, dict_var, covars, f_spare_input)

        # spare apply
        list_spare = []
        for SPARETYPE in spare_types:
            mdl = os.path.join(bdir, dict_config[f"model_SPARE-{SPARETYPE}"])
            f_spare = os.path.join(out_dir, f"{dset_name}_SPARE_{SPARETYPE}.csv")
            utilw.apply_spare(f_spare_input, mdl, SPARETYPE, f_spare)
            list_spare.append(f_spare)

        # merge spare
        utilw.merge_dataframes_multi(f_spares, "MRID", list_spare)

    def step_surrealgan():
        # Select variables for SurrealGAN
        dict_var = "Code"
        covars = "MRID,Age,Sex,DLICV"
        utilw.select_vars(f_comb, dict_csv, dict_var, covars, f_surrealgan_input)

        # Apply SurrealGAN index prediction
        utilw.surrealgan_scores(f_comb, dict_csv, f_surrealgan)
        
    def step_centiles():
        # Normalize ROIs
        icv_var = "MUSE_702"
        roi_pref = "MUSE"
        suffix = "NONE"
        utilw.corr_icv(f_combat_wcovar, f_raw, 'normICV', icv_var, roi_pref, suffix, f_combat_normicv)
        
        # Merge covars to normalized ROIs
        key_var = "MRID"
        utilw.merge_dataframes(input_demog, f_combat_normicv, key_var, f_combat_normicv_wcover)

        # find subject centile values
        cent_csv = '/home/guraylab/GitHub/CBICA/NiChart_Project/resources/centiles/istag_centiles_CN_ICV_Corrected.csv'
        utilw.calc_subject_centiles(f_combat_normicv_wcover, cent_csv, list_rois, f_centiles)

    def step_combine():
        # Merge all
        #utilw.combine_all(
            #f_results, [input_demog, rois_sel, f_raw, f_combat_normicv, f_combat_wderived, f_spares]
        #)
        #utilw.combine_demog_hroi_ml(
            #f_results, [input_demog, rois_sel, f_raw, f_combat_wderived, f_spares, f_surrealgan]
        #)
        utilw.combine_demog_hroi_ml_cent(
            f_results, [input_demog, rois_sel, f_raw, f_combat_wderived, f_centiles, f_spares, f_surrealgan]
        )

    bdir = os.path.join(root_dir, "src", "workflow")

    # Read dict_config vars and lists
    dset_name = dict_config["dset_name"]
    input_rois = dict_config["input_rois"]
    input_demog = dict_config["input_demog"]
    dir_output = dict_config["dir_output"]
    derived_rois = os.path.join(bdir, dict_config["list_derived_ROIs"])
    list_rois = os.path.join(bdir, dict_config["list_ROIs_all"])
    rois_single = os.path.join(bdir, dict_config["list_ROIs_single"])
    rois_sel = os.path.join(bdir, dict_config["list_ROIs_all"])
    spare_types = dict_config["SPARE_types"]
    model_combat = os.path.join(bdir, dict_config["model_combat"])
    # seg_types = dict_config["seg_types"]
    corr_type = dict_config["corr_type"]
    dict_csv = os.path.join("src", "workflow", rois_single)

    # Set output file name
    # OUT_FILE = f"{dir_output}/{dset_name}_DLMUSE+MLScores.csv"

    print(dict_config)

    out_dir = os.path.join(dir_output, "working_dir")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    f_raw = os.path.join(out_dir, f"{dset_name}_raw.csv")
    f_comb = os.path.join(out_dir, f"{dset_name}_comb.csv")
    f_combat_wderived = os.path.join(out_dir, f"{dset_name}_COMBAT_all.csv")
    f_combat_wcovar = os.path.join(out_dir, f"{dset_name}_COMBAT_withcovar.csv")
    f_spare_input = os.path.join(out_dir, f"{dset_name}_spare_input.csv")
    f_spares = os.path.join(out_dir, f"{dset_name}_SPARE-ALL.csv")
    f_surrealgan_input = os.path.join(out_dir, f"{dset_name}_surrealgan_input.csv")
    f_surrealgan = os.path.join(out_dir, f"{dset_name}_SurrealGAN.csv")
    f_combat_normicv = os.path.join(out_dir, f"{dset_name}_COMBAT_normICV.csv")
    f_combat_normicv_wcover = os.path.join(out_dir, f"{dset_name}_COMBAT_normICV_withcovar.csv")
    f_centiles = os.path.join(out_dir, f"{dset_name}_COMBAT_normICV_centiles.csv")
    f_results = os.path.join(dir_output, f"{dset_name}_DLMUSE+MLScores.csv")

    step_combat()
    step_spare()
    step_surrealgan()
    step_centiles()
    step_combine()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir", help="Provide the path to root for the script", required=True
    )
    parser.add_argument("--run_dir", help="Provide the path to script", required=True)
    parser.add_argument(
        "--dset_name", help="Provide a name for your dataset", required=True
    )
    parser.add_argument(
        "--input_rois", help="Provide input csv name with ROIs", required=True
    )
    parser.add_argument(
        "--input_demog",
        help="Provide input csv name with demographic info",
        required=True,
    )
    parser.add_argument(
        "--dir_output", help="Provide output folder name", required=True
    )

    options = parser.parse_args()

    # Create out dir
    if not os.path.exists(options.dir_output):
        os.makedirs(options.dir_output)

    # Change path
    os.chdir(options.run_dir)

    # Update dict_config file
    dict_config["dset_name"] = options.dset_name
    dict_config["input_rois"] = options.input_rois
    dict_config["input_demog"] = options.input_demog
    dict_config["dir_output"] = options.dir_output

    # Run workflow
    print("Running: wsmri")
    run_workflow(options.root_dir, dict_config)

    print("Workflow complete! Output file:", options.dir_output)
