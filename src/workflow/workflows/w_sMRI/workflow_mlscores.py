## Import packages
import pandas as pd
import json
import os
import utils_wsmri as utilw
import argparse

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
    "SPARE_types": ["AD", "Age", "Diabetes", "Hyperlipidemia", "Hypertension", "Obesity", "Smoking"],
    "seg_types": ["DLMUSE"]
}

def run_workflow(root_dir, dict_config):

    bdir = os.path.join(root_dir, 'src', 'workflow')

    # Read dict_config vars and lists
    dset_name = dict_config["dset_name"]
    input_rois = dict_config["input_rois"]
    input_demog = dict_config["input_demog"]
    dir_output = dict_config["dir_output"]
    derived_rois = os.path.join(bdir, dict_config["list_derived_ROIs"])
    list_rois = os.path.join(bdir, dict_config["list_ROIs_all"])
    rois_single = os.path.join(bdir, dict_config["list_ROIs_single"])
    rois_primary = os.path.join(bdir, dict_config["list_ROIs_primary"])
    spare_types = dict_config["SPARE_types"]
    model_combat = os.path.join(bdir, dict_config["model_combat"])
    seg_types = dict_config["seg_types"]
    corr_type = dict_config["corr_type"]

    ## Set output file name
    OUT_FILE = f"{dir_output}/{dset_name}_DLMUSE+MLScores.csv"

    print(dict_config)

    # Rename MUSE roi indices to roi codes
    out_tmp = os.path.join(dir_output, 'working_dir', 'out_rois')
    out_csv = os.path.join(out_tmp, f"{dset_name}_raw.csv")
    if not os.path.exists(out_tmp):
        os.makedirs(out_tmp)
    utilw.rename_df_columns(input_rois, list_rois, 'Index', 'Code', out_csv)

    # Normalize ROIs. Values are scaled either by a constant factor (NormICV) or 100 (PercICV)
    in_csv = out_csv
    icv_var = 'MUSE_702'
    exclude_vars = 'MRID'
    suffix = 'NONE'
    out_tmp = os.path.join(dir_output, 'working_dir', 'out_rois')
    if not os.path.exists(out_tmp):
        os.makedirs(out_tmp)
    out_csv = os.path.join(out_tmp, f"{dset_name}_{corr_type}.csv")
    utilw.corr_icv(in_csv, corr_type, icv_var, exclude_vars, suffix, out_csv)

    #Merge covars to ROIs
    covar=f"{input_demog}",
    roi=f"{dir_output}/working_dir/out_rois/{dset_name}_raw.csv",
    key_var = 'MRID'
    out = f"{dir_output}/working_dir/combined/{dset_name}_raw.csv"
    python src/workflow/utils/generic/util_merge_dfs.py covar roi key_var out

    ## Select variables for harmonization
    #in_csv=f"{dir_output}/working_dir/combined/{dset_name}_raw.csv",
    #dict_csv=f"src/workflow/{rois_single}"
    #dict_var = 'Code',
    #covars = 'MRID,Age,Sex,SITE,MUSE_702',
    #out = f"{dir_output}/working_dir/sel_vars/{dset_name}_raw.csv"
    #python src/workflow/utils/generic/util_select_vars.py {input} {params} {output}

    ## Check that sample has age range consistent with the model
    #f"{dir_output}/working_dir/sel_vars/{dset_name}_raw.csv",
    #var_name='Age',
    #min_val='50',
    #max_val='95',
    #"{dir_output}/working_dir/filtered_data/{dset_name}_raw.csv"
    #python src/workflow/utils/generic/util_filter_num_var.py {input} {params} {output}

    ## Apply combat
    #data=f"{dir_output}/working_dir/filtered_data/{dset_name}_raw.csv",
    #mdl=f"src/workflow/{model_combat}"
    #f"{dir_output}/working_dir/out_combat/{dset_name}_COMBAT_single.csv"
    #python combat_prep_out(in_csv, key_var, suffix, out_csv)

    ## Calculate derived ROIs from harmonized data
    #in_csv=f"{dir_output}/working_dir/out_combat/{dset_name}_COMBAT_single.csv",
    #dict=f"src/workflow/{derived_rois}"
    #key_var='MRID',
    #roi_prefix='MUSE_'
    #f"{dir_output}/working_dir/out_combat/{dset_name}_COMBAT_all.csv"
    #python src/workflow/utils/generic/util_combine_MUSE_rois.py {input} {params} {output}

    ## Merge covars to ROIs
    #covar=f"{input_demog}",
    #roi=f"{dir_output}/working_dir/out_combat/{dset_name}_COMBAT_single.csv"
    #key_var = 'MRID'
    #f"{dir_output}/working_dir/spare/{dset_name}_COMBAT_withcovar.csv"
    #python src/workflow/utils/generic/util_merge_dfs.py {input} {params} {output}"

    ## Select variables for harmonization
    #in_csv=f"{dir_output}/working_dir/spare/{dset_name}_COMBAT_withcovar.csv"
    #dict_csv=f"src/workflow/{rois_single}"
    #dict_var = 'Code'
    #covars ='MRID,Age,Sex,DLICV',
    #f"{dir_output}/working_dir/spare/{dset_name}_COMBAT.csv"
    #python src/workflow/utils/generic/util_select_vars.py {input} {params} {output}

    ##def get_spare_model(wildcards):
        ##model_name = dict_config["model_SPARE-" + wildcards.stype]
        ##path_spare = "src/workflow/" + model_name
        ##return path_spare

    ## spare apply
    #data=f"{dir_output}/working_dir/spare/{dset_name}_COMBAT.csv",
    #mdl=get_spare_model
    #f"{dir_output}/working_dir/out_spare/{dset_name}_COMBAT_SPARE-{{stype}}.csv"
    #"bash src/workflow/utils/spare/util_spare_test.sh {input} {wildcards.stype} {output}"

    ### get_spare_results(wildcards):
        ##data_spare=expand(f"{dir_output}/working_dir/out_spare/{dset_name}_COMBAT_SPARE-{{stype}}.csv", stype = spare_types)
        ##return data_spare

    ## spare_combine
    #get_spare_results
    #csv=f"{dir_output}/working_dir/out_spare/{dset_name}_COMBAT_SPARE-Scores.csv"
    #python src/workflow/utils/generic/util_merge_dfs_multi.py {output} MRID {input}

    ## Merge demog data to DLMUSE
    #demog=f"{input_demog}",
    #rois=f"src/workflow/{rois_primary}",
    #out_raw=f"{dir_output}/working_dir/out_rois/{dset_name}_raw.csv",
    #out_corr=f"{dir_output}/working_dir/out_rois/{dset_name}_{corr_type}.csv",
    #out_harm=f"{dir_output}/working_dir/out_combat/{dset_name}_COMBAT_all.csv",
    #out_spare=f"{dir_output}/working_dir/out_spare/{dset_name}_COMBAT_SPARE-Scores.csv"
    #f"{dir_output}/{dset_name}_DLMUSE+MLScores.csv"
    #key_var = 'MRID'
    #python src/workflow/utils/generic/util_combine_all.py {output} {input}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir", help="Provide the path to root for the script", required=True
    )
    parser.add_argument(
        "--run_dir", help="Provide the path to script", required=True
    )
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
    dict_config['dset_name'] = options.dset_name
    dict_config['input_rois'] = options.input_rois
    dict_config['input_demog'] = options.input_demog
    dict_config['dir_output'] = options.dir_output

    # Run workflow
    print(f"Running: wsmri")
    run_workflow(options.root_dir, dict_config)

    print("Workflow complete! Output file:", options.dir_output)


