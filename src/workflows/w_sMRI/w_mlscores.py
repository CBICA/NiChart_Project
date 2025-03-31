# Import packages
import csv as csv
import os

import numpy as np
import pandas as pd
from stqdm import stqdm

def check_input(
    in_csv: str,
    in_demog: str,
) -> list:
    # Read input csv
    try:
        df_in = pd.read_csv(in_csv, dtype={"MRID": str})
    except:
        return [-1, 'Could not read data file']
    
    # Read input csv
    try:
        df_demog = pd.read_csv(in_demog, dtype={"MRID": str})
    except:
        return [-1, 'Could not read demographics file']
    
    # Check required columns
    for sel_col in ['MRID']:
        if sel_col not in df_in.columns:
            return [-1, f'Required column missing in data file: {sel_col}']
    for sel_col in ['MRID', 'Age', 'Sex']:
        if sel_col not in df_demog.columns:
            return [-1, f'Required column missing in demographics file: {sel_col}']
    
    # Merge data files
    try:
        df = df_demog.merge(df_in, on='MRID')
    except:
        return [-1, 'Could not merge data and demographics files']
    
    # Check data size
    if df.shape[0] == 0:
        return [-1, 'No matching MRIDs found between data and demographics files']
    
    return [0, 'Data verification Successful']
    
    
    
def combine_rois(
    df_in: pd.DataFrame,
    csv_roi_dict: str,
) -> pd.DataFrame:
    """
    Calculates a dataframe with the volumes of derived + single rois.
    """
    key_var = "MRID"
    roi_prefix = "MUSE_"
    # Read derived roi map file to a dictionary
    dict_roi = {}
    with open(csv_roi_dict) as roi_map:
        reader = csv.reader(roi_map, delimiter=",")
        for row in reader:
            key = roi_prefix + str(row[0])
            val = [roi_prefix + str(x) for x in row[2:]]
            dict_roi[key] = val

    # Create derived roi df
    label_names = list(dict_roi.keys())
    df_out = df_in[[key_var]]
    df_out = df_out.reindex(columns=[key_var] + label_names)

    # Calculate volumes for derived rois
    for i, key in enumerate(dict_roi):
        key_vals = dict_roi[key]
        try:
            if key in df_in.columns:
                df_out[key] = df_in[
                    key
                ]  # If ROI is already in df, do not calculate it from parts
                print("WARNING:  Skip derived ROI, already in data: " + key)
            else:
                df_out[key] = df_in[key_vals].sum(
                    axis=1
                )  # If not, calculate it (sum of single ROIs in the dict)
        except:
            df_out = df_out.drop(columns=key)
            print("WARNING:  Skip derived ROI with missing components: " + key)

    return df_out


def calc_subject_centiles(
    df_in: pd.DataFrame, df_cent: pd.DataFrame, df_dict: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculates subject specific centile values
    """
    # Rename centiles roi names
    rdict = dict(zip(df_dict["Name"], df_dict["Code"]))
    df_cent["VarName"] = df_cent["VarName"].replace(rdict)

    cent = df_cent.columns[2:].str.replace("centile_", "").astype(int).values

    # Get age bin
    df_in["Age"] = df_in.Age.round(0)
    df_in.loc[df_in["Age"] > df_cent.Age.max(), "Age"] = df_cent.Age.max()
    df_in.loc[df_in["Age"] < df_cent.Age.min(), "Age"] = df_cent.Age.min()

    # Find ROIs
    sel_vars = df_in.columns[df_in.columns.isin(df_cent.VarName.unique())].tolist()

    # For each subject find the centile value of each roi
    cent_subj_all = np.zeros([df_in.shape[0], len(sel_vars)])
    for i, tmp_ind in enumerate(df_in.index):
        df_subj = df_in.loc[[tmp_ind]]
        df_cent_sel = df_cent[df_cent.Age == df_subj.Age.values[0]]

        for j, tmp_var in enumerate(sel_vars):
            # Get centile values
            vals_cent = df_cent_sel[df_cent_sel.VarName == tmp_var].values[0][2:]

            # Get subject value
            sval = df_subj[tmp_var].values[0]

            # Find nearest x values
            sval = np.min([vals_cent[-1], np.max([vals_cent[0], sval])])
            ind1 = np.where(sval <= vals_cent)[0][0] - 1
            if ind1 == -1:
                ind1 = 0
            ind2 = ind1 + 1

            # Calculate slope
            slope = (cent[ind2] - cent[ind1]) / (vals_cent[ind2] - vals_cent[ind1])

            # Estimate subj centile
            cent_subj_all[i, j] = cent[ind1] + slope * (sval - vals_cent[ind1])

    # Create and save output data
    df_out = pd.DataFrame(columns=sel_vars, data=cent_subj_all)
    df_out = pd.concat([df_in[["MRID"]], df_out], axis=1)

    return df_out


def run_workflow(
    dset_name: str,
    bdir: str,
    in_csv: str,
    in_demog: str,
    out_dir: str,
) -> None:
    # Fixed params
    key_var = "MRID"
    min_age = 50
    max_age = 95
    suff_combat = "_HARM"
    spare_types = ["AD", "Age", "Hypertension", "Diabetes", "Hyperlipidemia", "Obesity", "Smoking"]
    icv_ref_val = 1430000
    cent_csv = os.path.join(
        bdir, "resources", "centiles", "istag_centiles_CN_ICV_Corrected.csv"
    )
    csv_muse_all = os.path.join(
        bdir, "src", "workflows", "w_sMRI", "lists", "list_MUSE_all.csv"
    )
    csv_muse_single = os.path.join(
        bdir, "src", "workflows", "w_sMRI", "lists", "list_MUSE_single.csv"
    )
    csv_muse_derived = os.path.join(
        bdir, "src", "workflows", "w_sMRI", "lists", "list_MUSE_mapping_derived.csv"
    )
    model_combat = os.path.join(
        bdir,
        "src",
        "workflows",
        "w_sMRI",
        "models",
        "vISTAG1",
        "COMBAT",
        "combined_DLMUSE_raw_COMBATModel.pkl.gz",
    )
    spare_dir = os.path.join(
        bdir, "src", "workflows", "w_sMRI", "models", "vISTAG1", "SPARE"
    )
    spare_pref = "combined_DLMUSE_raw_COMBAT_SPARE-"
    spare_suff = "_Model.pkl.gz"

    # Print args
    print(
        f"About to run: run_workflow {dset_name} {bdir} {in_csv} {in_demog} {out_dir}"
    )

    def step_combat() -> None:
        # Read input
        f_in = os.path.join(out_wdir, f"{dset_name}_rois_init.csv")
        df_in = pd.read_csv(f_in, dtype={"MRID": str})

        # Check SITE column
        if 'SITE' not in df_in.columns:
            df_in['SITE'] = 'SITE1'

        # Select variables for harmonization
        muse_vars = df_in.columns[df_in.columns.str.contains("MUSE")].tolist()
        other_vars = ["MRID", "Age", "Sex", "SITE", "DLICV"]
        df_out = df_in[other_vars + muse_vars]

        # Check that sample has age range consistent with the model
        df_out = df_out[df_out["Age"] > min_age]
        df_out = df_out[df_out["Age"] < max_age]

        # Save combat input
        f_combat_in = os.path.join(out_wdir, f"{dset_name}_combat_in.csv")
        df_out.to_csv(f_combat_in, index=False)

        # Apply combat
        mdl_combat = os.path.join("src", "workflow", model_combat)
        f_combat_out = os.path.join(out_wdir, f"{dset_name}_combat_init.csv")
        os.system(
            f"neuroharm -a apply -i {f_combat_in} -m {mdl_combat} -u {f_combat_out}"
        )

        # Edit combat output (remove non mri columns and suffix combat)
        df_combat = pd.read_csv(f_combat_out, dtype={"MRID": str})
        df_combat.columns = df_combat.columns.astype(str)
        sel_vars = [key_var] + df_combat.columns[
            df_combat.columns.str.contains(suff_combat)
        ].tolist()
        df_combat = df_combat[sel_vars]
        df_combat.columns = df_combat.columns.str.replace(suff_combat, "")

        # Add derived rois
        df_combat = combine_rois(df_combat, csv_muse_derived)

        # Merge covars to harmonized ROIs
        df_combat = df_demog.merge(df_combat, on=key_var)

        # Change DLICV to ICV
        df_combat = df_combat.rename(columns={'DLICV':'ICV'})

        # Write out file
        f_combat_out = os.path.join(out_wdir, f"{dset_name}_combat.csv")
        df_combat.to_csv(f_combat_out, index=False)

    def step_centiles() -> None:
        # Read input
        f_in = os.path.join(out_wdir, f"{dset_name}_combat.csv")
        df_in = pd.read_csv(f_in, dtype={"MRID": str})

        # Normalize ROIs
        df_icvcorr = df_in.copy()
        var_muse = df_icvcorr.columns[df_icvcorr.columns.str.contains("MUSE")]
        df_tmp = df_icvcorr[var_muse]
        df_tmp = df_tmp.div(df_in["ICV"], axis=0) * icv_ref_val
        # df_tmp=df_tmp.div(df_icvcorr['ICV'], axis=0) * 100
        df_icvcorr[var_muse] = df_tmp.values

        # Calculate centiles
        df_cent = pd.read_csv(cent_csv)
        df_centiles = calc_subject_centiles(df_icvcorr, df_cent, df_roidict)

        # Write out files
        f_icvcorr = os.path.join(out_wdir, f"{dset_name}_combat_icvcorr.csv")
        df_icvcorr.to_csv(f_icvcorr, index=False)
        f_centiles = os.path.join(out_wdir, f"{dset_name}_combat_icvcorr_centiles.csv")
        df_centiles.to_csv(f_centiles, index=False)

    def step_spare() -> None:
        # Read input
        f_in = os.path.join(out_wdir, f"{dset_name}_combat.csv")
        df_in = pd.read_csv(f_in, dtype={"MRID": str})

        # Apply spare
        df_spare = df_in[["MRID"]]
        for spare_type in spare_types:
            spare_mdl = os.path.join(spare_dir, f"{spare_pref}{spare_type}{spare_suff}")
            f_spare_out = os.path.join(
                out_wdir, f"{dset_name}_combat_spare_{spare_type}.csv"
            )
            os.system(f"spare_score -a test -i {f_in} -m {spare_mdl} -o {f_spare_out}")

            # Change column name for the spare output
            df = pd.read_csv(f_spare_out)
            df = df[df.columns[0:2]]
            df.columns = ["MRID", f"SPARE{spare_type}"]
            df_spare = df_spare.merge(df)

        # Write out file with all spare scores
        f_spare_out = os.path.join(out_wdir, f"{dset_name}_combat_spare-all.csv")
        df_spare.to_csv(f_spare_out, index=False)

    def step_sgan() -> None:
        # Read input
        f_in = os.path.join(out_wdir, f"{dset_name}_rois_init.csv")
        df_in = pd.read_csv(f_in, dtype={"MRID": str})

        # Select input
        sel_covars = ["MRID", "Age", "Sex", "DLICV"]
        sel_vars = sel_covars + list_muse_single
        df_sel = df_in[sel_vars]
        f_sgan_in = os.path.join(out_wdir, f"{dset_name}_sgan_in.csv")
        df_sel.to_csv(f_sgan_in, index=False)

        # Run prediction
        f_sgan_out = os.path.join(out_wdir, f"{dset_name}_sgan_init.csv")
        cmd = f"PredCRD -i {f_sgan_in} -o {f_sgan_out}"
        print(f"About to run {cmd}")
        os.system(cmd)

        # Edit columns
        df_sgan = pd.read_csv(f_sgan_out)
        df_sgan.columns = ["MRID"] + df_sgan.add_prefix("SurrealGAN_").columns[
            1:
        ].tolist()
        f_sgan = os.path.join(out_wdir, f"{dset_name}_sgan.csv")
        df_sgan.to_csv(f_sgan, index=False)

    def step_cclnmf() -> None:
        # Read input
        f_in = os.path.join(out_wdir, f"{dset_name}_rois_init.csv")
        df_in = pd.read_csv(f_in, dtype={"MRID": str})

        # Select input
        sel_demog_vars = ["MRID", "Age", "Sex", "DLICV"]
        sel_vars = ["MRID"] + list_muse_single
        df_self = df_in[sel_vars]
        df_demog_sel = df_in[sel_demog_vars]
        f_cclnmf_in = os.path.join(out_wdir, f"{dset_name}_cclnmf_in.csv")
        df_sel.to_csv(f_cclnmf_in, index=False)
        f_cclnmf_demog_in = os.path.join(out_wdir, f"{dset_name}_cclnmf_demographics.csv")
        df_demog_sel.to_csv(f_cclnmf_demog_in, index=False)

        # Run prediction
        f_cclnmf_out = os.path.join(out_wdir, f"{dset_name}_cclnmf_init.csv")
        cmd = f"ccl_nmf_prediction -i {f_cclnmf_in} -o {f_cclnmf_demog_in} -o {f_cclnmf_out}"
        print(f"About to run {cmd}")
        os.system(cmd)

        # Edit columns
        df_cclnmf = pd.read_csv(f_cclnmf_out)
        df_cclnmf_columns = ["MRID"] + df_cclnmf.add_prefix("CCLNMF_").columns[
            1:
        ].tolist()

        # Export to csv
        f_cclnmf = os.path.join(out_wdir, f"{dset_name}_cclnmf.csv")
        df_cclnmf.to_csv(f_cclnmf, index=False)

    # Make out dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_wdir = os.path.join(out_dir, "working_dir")
    if not os.path.exists(out_wdir):
        os.makedirs(out_wdir)

    # Read roi lists
    df_tmp = pd.read_csv(csv_muse_single)
    list_muse_single = df_tmp.Code.tolist()

    # Read data
    df = pd.read_csv(in_csv, dtype={"MRID": str})
    df.columns = df.columns.astype(str)

    # Rename ROIs
    df_roidict = pd.read_csv(csv_muse_all)
    df_roidict.Index = df_roidict.Index.astype(str)
    vdict = df_roidict.set_index("Index")["Code"].to_dict()
    df = df.rename(columns=vdict)

    # Keep DLICV in a separate df
    df_icv = df[["MRID", "MUSE_702"]]
    df_icv.columns = ["MRID", "DLICV"]
    df = df.drop(columns=["MUSE_702"])

    # Add DLICV to demog
    df_demog = pd.read_csv(in_demog, dtype={"MRID": str})
    df_demog = df_demog.merge(df_icv, on="MRID")

    # Add covars
    df_raw = df_demog.merge(df, on=key_var)
    f_raw = os.path.join(out_wdir, f"{dset_name}_rois_init.csv")
    df_raw.to_csv(f_raw, index=False)

    list_func = [step_combat, step_centiles, step_spare, step_sgan, step_cclnmf]
    # list_fnames = [
    # "COMBAT harmonization",
    # "Centile calculation",
    # "SPARE index calculation",
    # "SurrealGAN index calculation",
    # "CCL_NMF calculation",
    # ]

    for i, sel_func in stqdm(
        enumerate(list_func),
        desc="Running step ...",
        total=len(list_func),
    ):
        sel_func()

    # Combine results

    # Rename roi names and merge dfs
    f_combat = os.path.join(out_wdir, f"{dset_name}_combat.csv")
    df_out = pd.read_csv(f_combat, dtype={"MRID": str})
    df_out = df_out.rename(columns=dict(zip(df_roidict.Code, df_roidict.Name)))
    df_out = df_out.loc[:,~df_out.columns.duplicated()]
    df_out = df_out.rename(columns={'DLICV':'ICV'})

    f_centiles = os.path.join(out_wdir, f"{dset_name}_combat_icvcorr_centiles.csv")
    df_centiles = pd.read_csv(f_centiles, dtype={"MRID": str})
    df_tmp = df_centiles[
        ["MRID"]
        + df_centiles.columns[df_centiles.columns.str.contains("MUSE")].tolist()
    ]
    df_tmp = df_tmp.rename(columns={'DLICV':'ICV'})
    df_tmp = df_tmp.rename(columns=dict(zip(df_roidict.Code, df_roidict.Name)))
    df_tmp = df_tmp.loc[:,~df_tmp.columns.duplicated()]
    df_out = df_out.merge(df_tmp, on="MRID", suffixes=["", "_centiles"])

    f_spare = os.path.join(out_wdir, f"{dset_name}_combat_spare-all.csv")
    df_spare = pd.read_csv(f_spare, dtype={"MRID": str})
    df_out = df_out.merge(df_spare, on="MRID")

    f_sgan = os.path.join(out_wdir, f"{dset_name}_sgan.csv")
    df_sgan = pd.read_csv(f_sgan, dtype={"MRID": str})
    df_out = df_out.merge(df_sgan, on="MRID")

    # Write out file
    f_results = os.path.join(out_dir, f"{dset_name}_DLMUSE+MLScores.csv")
    df_out.to_csv(f_results, index=False)


def run_workflow_noharmonization(
    dset_name: str,
    bdir: str,
    in_csv: str,
    in_demog: str,
    out_dir: str,
) -> None:
    # Fixed params
    key_var = "MRID"
    spare_types = ["AD", "Age"]
    icv_ref_val = 1430000
    cent_csv = os.path.join(
        bdir, "resources", "centiles", "istag_centiles_CN_ICV_Corrected.csv"
    )
    csv_muse_all = os.path.join(
        bdir, "src", "workflows", "w_sMRI", "lists", "list_MUSE_all.csv"
    )
    csv_muse_single = os.path.join(
        bdir, "src", "workflows", "w_sMRI", "lists", "list_MUSE_single.csv"
    )
    # csv_muse_derived = os.path.join(
    # bdir, "src", "workflows", "w_sMRI", "lists", "list_MUSE_mapping_derived.csv"
    # )
    spare_dir = os.path.join(
        bdir, "src", "workflows", "w_sMRI", "models", "vISTAG1", "SPARE"
    )
    spare_pref = "combined_DLMUSE_raw_COMBAT_SPARE-"
    spare_suff = "_Model.pkl.gz"

    # Print args
    print(
        f"About to run: run_workflow {dset_name} {bdir} {in_csv} {in_demog} {out_dir}"
    )
    

    def step_centiles() -> None:
        # Read input
        f_in = os.path.join(out_wdir, f"{dset_name}_rois_init.csv")
        df_in = pd.read_csv(f_in, dtype={"MRID": str})

        # Normalize ROIs
        df_icvcorr = df_in.copy()
        var_muse = df_icvcorr.columns[df_icvcorr.columns.str.contains("MUSE")]
        df_tmp = df_icvcorr[var_muse]
        df_tmp = df_tmp.div(df_in["DLICV"], axis=0) * icv_ref_val
        # df_tmp=df_tmp.div(df_icvcorr['DLICV'], axis=0) * 100
        df_icvcorr[var_muse] = df_tmp.values

        # Calculate centiles
        df_cent = pd.read_csv(cent_csv)
        df_centiles = calc_subject_centiles(df_icvcorr, df_cent, df_roidict)

        # Write out files
        f_icvcorr = os.path.join(out_wdir, f"{dset_name}_icvcorr.csv")
        df_icvcorr.to_csv(f_icvcorr, index=False)
        f_centiles = os.path.join(out_wdir, f"{dset_name}_icvcorr_centiles.csv")
        df_centiles.to_csv(f_centiles, index=False)

    def step_spare() -> None:
        # Read input
        f_in = os.path.join(out_wdir, f"{dset_name}_rois_init.csv")
        df_in = pd.read_csv(f_in, dtype={"MRID": str})

        # Apply spare
        df_spare = df_in[["MRID"]]
        for spare_type in spare_types:
            spare_mdl = os.path.join(spare_dir, f"{spare_pref}{spare_type}{spare_suff}")
            f_spare_out = os.path.join(out_wdir, f"{dset_name}_spare_{spare_type}.csv")
            os.system(f"spare_score -a test -i {f_in} -m {spare_mdl} -o {f_spare_out}")

            # Change column name for the spare output
            df = pd.read_csv(f_spare_out)
            df = df[df.columns[0:2]]
            df.columns = ["MRID", f"SPARE{spare_type}"]
            df_spare = df_spare.merge(df)

        # Write out file with all spare scores
        f_spare_out = os.path.join(out_wdir, f"{dset_name}_spare-all.csv")
        df_spare.to_csv(f_spare_out, index=False)

    def step_sgan() -> None:
        # Read input
        f_in = os.path.join(out_wdir, f"{dset_name}_rois_init.csv")
        df_in = pd.read_csv(f_in, dtype={"MRID": str})

        # Select input
        sel_covars = ["MRID", "Age", "Sex", "DLICV"]
        sel_vars = sel_covars + list_muse_single
        df_sel = df_in[sel_vars]
        f_sgan_in = os.path.join(out_wdir, f"{dset_name}_sgan_in.csv")
        df_sel.to_csv(f_sgan_in, index=False)

        # Run prediction
        f_sgan_out = os.path.join(out_wdir, f"{dset_name}_sgan_init.csv")
        cmd = f"PredCRD -i {f_sgan_in} -o {f_sgan_out}"
        print(f"About to run {cmd}")
        os.system(cmd)

        # Edit columns
        df_sgan = pd.read_csv(f_sgan_out)
        df_sgan.columns = ["MRID"] + df_sgan.add_prefix("SurrealGAN_").columns[
            1:
        ].tolist()
        f_sgan = os.path.join(out_wdir, f"{dset_name}_sgan.csv")
        df_sgan.to_csv(f_sgan, index=False)
    
    def step_cclnmf() -> None:
        # Read input
        f_in = os.path.join(out_wdir, f"{dset_name}_rois_init.csv")
        df_in = pd.read_csv(f_in, dtype={"MRID": str})

        # Select input
        sel_demog_vars = ["MRID", "Age", "Sex", "DLICV"]
        sel_vars = ["MRID"] + list_muse_single
        df_self = df_in[sel_vars]
        df_demog_sel = df_in[sel_demog_vars]
        f_cclnmf_in = os.path.join(out_wdir, f"{dset_name}_cclnmf_in.csv")
        df_sel.to_csv(f_cclnmf_in, index=False)
        f_cclnmf_demog_in = os.path.join(out_wdir, f"{dset_name}_cclnmf_demographics.csv")
        df_demog_sel.to_csv(f_cclnmf_demog_in, index=False)

        # Run prediction
        f_cclnmf_out = os.path.join(out_wdir, f"{dset_name}_cclnmf_init.csv")
        cmd = f"ccl_nmf_prediction -i {f_cclnmf_in} -o {f_cclnmf_demog_in} -o {f_cclnmf_out}"
        print(f"About to run {cmd}")
        os.system(cmd)

        # Edit columns
        df_cclnmf = pd.read_csv(f_cclnmf_out)
        df_cclnmf_columns = ["MRID"] + df_cclnmf.add_prefix("CCLNMF_").columns[
            1:
        ].tolist()

        # Export to csv
        f_cclnmf = os.path.join(out_wdir, f"{dset_name}_cclnmf.csv")
        df_cclnmf.to_csv(f_cclnmf, index=False)
    

    # Make out dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_wdir = os.path.join(out_dir, "working_dir")
    if not os.path.exists(out_wdir):
        os.makedirs(out_wdir)

    # Read roi lists
    df_tmp = pd.read_csv(csv_muse_single)
    list_muse_single = df_tmp.Code.tolist()

    # Read data
    df = pd.read_csv(in_csv, dtype={"MRID": str})
    df.columns = df.columns.astype(str)

    # Rename ROIs
    df_roidict = pd.read_csv(csv_muse_all)
    df_roidict.Index = df_roidict.Index.astype(str)
    vdict = df_roidict.set_index("Index")["Code"].to_dict()
    df = df.rename(columns=vdict)

    # Keep DLICV in a separate df
    df_icv = df[["MRID", "MUSE_702"]]
    df_icv.columns = ["MRID", "DLICV"]
    df = df.drop(columns=["MUSE_702"])

    # Add DLICV to demog
    df_demog = pd.read_csv(in_demog, dtype={"MRID": str})
    df_demog = df_demog.merge(df_icv, on="MRID")

    # Add covars
    df_raw = df_demog.merge(df, on=key_var)
    f_raw = os.path.join(out_wdir, f"{dset_name}_rois_init.csv")
    df_raw.to_csv(f_raw, index=False)

    list_func = [step_centiles, step_spare, step_sgan, step_cclnmf]
    # list_fnames = [
    # "Centile calculation",
    # "SPARE index calculation",
    # "SurrealGAN index calculation",
    # "CCL_NMF calculation",
    # ]

    for i, sel_func in stqdm(
        enumerate(list_func),
        desc="Running step ...",
        total=len(list_func),
    ):
        sel_func()

    # Combine results

    # Rename roi names and merge dfs
    f_raw = os.path.join(out_wdir, f"{dset_name}_rois_init.csv")
    df_out = pd.read_csv(f_raw, dtype={"MRID": str})
    df_out = df_out.rename(columns=dict(zip(df_roidict.Code, df_roidict.Name)))
    df_out = df_out.loc[:,~df_out.columns.duplicated()]
    df_out = df_out.rename(columns={'DLICV':'ICV'})

    f_centiles = os.path.join(out_wdir, f"{dset_name}_icvcorr_centiles.csv")
    df_centiles = pd.read_csv(f_centiles, dtype={"MRID": str})
    df_tmp = df_centiles[
        ["MRID"]
        + df_centiles.columns[df_centiles.columns.str.contains("MUSE")].tolist()
    ]
    df_tmp = df_tmp.rename(columns=dict(zip(df_roidict.Code, df_roidict.Name)))
    df_tmp = df_tmp.loc[:,~df_tmp.columns.duplicated()]
    df_tmp = df_tmp.rename(columns={'DLICV':'ICV'})    
    df_out = df_out.merge(df_tmp, on="MRID", suffixes=["", "_centiles"])

    f_spare = os.path.join(out_wdir, f"{dset_name}_spare-all.csv")
    df_spare = pd.read_csv(f_spare, dtype={"MRID": str})
    df_out = df_out.merge(df_spare, on="MRID")

    f_sgan = os.path.join(out_wdir, f"{dset_name}_sgan.csv")
    df_sgan = pd.read_csv(f_sgan, dtype={"MRID": str})
    df_out = df_out.merge(df_sgan, on="MRID")

    # Write out file
    f_results = os.path.join(out_dir, f"{dset_name}_DLMUSE+MLScores.csv")
    df_out.to_csv(f_results, index=False)
