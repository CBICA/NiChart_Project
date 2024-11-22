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
    "SPARE_types": ["AD", "Age"],
    "seg_types": ["DLMUSE"],
}


def run_workflow(root_dir: Any, dict_config: Any) -> None:

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

    # Set output file name
    # OUT_FILE = f"{dir_output}/{dset_name}_DLMUSE+MLScores.csv"

    print(dict_config)

    # Rename MUSE roi indices to roi codes
    out_dir = os.path.join(dir_output, "working_dir")
    f_raw = os.path.join(out_dir, f"{dset_name}_raw.csv")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    utilw.rename_df_columns(input_rois, list_rois, "Index", "Code", f_raw)

    # Normalize ROIs. Values are scaled either by a constant factor (NormICV) or 100 (PercICV)
    icv_var = "MUSE_702"
    exclude_vars = "MRID"
    suffix = "NONE"
    f_corr = os.path.join(out_dir, f"{dset_name}_{corr_type}.csv")
    utilw.corr_icv(f_raw, corr_type, icv_var, exclude_vars, suffix, f_corr)

    # Merge covars to ROIs
    key_var = "MRID"
    f_comb = os.path.join(out_dir, f"{dset_name}_comb.csv")
    utilw.merge_dataframes(input_demog, f_raw, key_var, f_comb)

    # Select variables for harmonization
    dict_csv = os.path.join("src", "workflow", rois_single)
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
    f_combat2 = os.path.join(out_dir, f"{dset_name}_COMBAT_all.csv")
    utilw.combine_rois(f_combat1, in_dict, f_combat2)

    # Merge covars to ROIs
    key_var = "MRID"
    f_combat3 = os.path.join(out_dir, f"{dset_name}_COMBAT_withcovar.csv")
    utilw.merge_dataframes(input_demog, f_combat2, key_var, f_combat3)

    # Select variables for harmonization
    dict_var = "Code"
    covars = "MRID,Age,Sex,DLICV"
    f_combat4 = os.path.join(out_dir, f"{dset_name}_COMBAT.csv")
    utilw.select_vars(f_combat3, dict_csv, dict_var, covars, f_combat4)

    # spare apply
    list_spare = []
    for SPARETYPE in spare_types:
        mdl = os.path.join(bdir, dict_config[f"model_SPARE-{SPARETYPE}"])
        f_spare = os.path.join(out_dir, f"{dset_name}_SPARE_{SPARETYPE}.csv")
        utilw.apply_spare(f_combat4, mdl, SPARETYPE, f_spare)
        list_spare.append(f_spare)

    # merge spare
    f_spares = os.path.join(out_dir, f"{dset_name}_SPARE-ALL.csv")
    utilw.merge_dataframes_multi(f_spares, "MRID", list_spare)

    ## Select variables for SurrealGAN
    #dict_var = "Code"
    #covars = "MRID,Age,Sex,DLICV"
    #f_surrealgan_input = os.path.join(out_dir, f"{dset_name}_SurrealGAN_input.csv")
    #utilw.select_vars(f_comb, dict_csv, dict_var, covars, f_surrealgan_input)

    ## Apply SurrealGAN index prediction
    #f_surrealgan = os.path.join(out_dir, f"{dset_name}_SurrealGAN.csv")
    #utilw.surrealgan_scores(f_comb, dict_csv, f_surrealgan)
    
    # Merge all
    f_all = os.path.join(dir_output, f"{dset_name}_DLMUSE+MLScores.csv")
    #utilw.combine_all(
        #f_all, [input_demog, rois_sel, f_raw, f_corr, f_combat2, f_spares]
    #)
    utilw.combine_demog_hroi_ml(
        f_all, [input_demog, rois_sel, f_raw, f_combat2, f_spares]
    )


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
