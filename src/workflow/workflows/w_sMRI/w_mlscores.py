# Import packages
import argparse
import os
from typing import Any
import pandas as pd
import csv as csv
import numpy as np

def combine_rois(
    df_in: str,
    csv_roi_dict: str,
) -> pd.DataFrame:
    """
    Calculates a dataframe with the volumes of derived + single rois.
    """
    roi_prefix='MUSE_'
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
    df_out = df_in[[key_var]].copy()
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

def calc_subject_centiles(df_in: str, df_cent: str, df_dict: str):
    """
    Calculates subject specific centile values
    """
    # Rename centiles roi names
    rdict = dict(zip(df_dict['Name'], df_dict['Code']))
    df_cent['VarName'] = df_cent['VarName'].replace(rdict)
    
    cent = df_cent.columns[2:].str.replace("centile_", "").astype(int).values
    
    # Get age bin
    df_in['Age'] = df_in.Age.round(0)
    df_in.loc[df_in['Age']>df_cent.Age.max(), 'Age'] = df_cent.Age.max()
    df_in.loc[df_in['Age']<df_cent.Age.min(), 'Age'] = df_cent.Age.min()

    # Find ROIs
    sel_vars = df_in.columns[df_in.columns.isin(df_cent.VarName.unique())].tolist()
    
    # For each subject find the centile value of each roi
    cent_subj_all = np.zeros([df_in.shape[0], len(sel_vars)])
    for i, tmp_ind in enumerate(df_in.index):
        df_subj = df_in.loc[[tmp_ind]]
        df_cent_sel = df_cent[df_cent.Age == df_subj.Age.values[0]]

        for j,tmp_var in enumerate(sel_vars):
            # Get centile values
            vals_cent = df_cent_sel[df_cent_sel.VarName==tmp_var].values[0][2:]

            # Get subject value
            sval = df_subj[tmp_var].values[0]

            # Find nearest x values
            sval = np.min([vals_cent[-1], np.max([vals_cent[0], sval])])
            ind1 = np.where(sval <= vals_cent)[0][0] -1
            if ind1 == -1:
                ind1 = 0
            ind2 = ind1 + 1

            # Calculate slope
            slope = (cent[ind2] - cent[ind1]) / (vals_cent[ind2] - vals_cent[ind1])

            # Estimate subj centile
            cent_subj_all[i,j] = cent[ind1] + slope * (sval - vals_cent[ind1])
            
    # Create and save output data
    df_out = pd.DataFrame(columns=sel_vars, data=cent_subj_all)
    df_out = pd.concat([df_in[['MRID']], df_out], axis=1)

    return df_out

def run_workflow(
    dset_name,
    in_csv,
    in_demog,
    out_dir
) -> None:
    # const def
    key_var="MRID"
    min_age=50
    max_age=95
    suff_combat='_HARM'
    spare_types=['AD', 'Age']
    icv_ref_val=1430000  
    cent_csv='/home/guraylab/GitHub/CBICA/NiChart_Project/resources/centiles/istag_centiles_CN_ICV_Corrected.csv'
    in_rois_single='/home/guraylab/GitHub/CBICA/NiChart_Project/src/workflow/MUSE/list_MUSE_single.csv'

    # Make out dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # List centiles

    # Read roi lists
    df_tmp=pd.read_csv(in_rois_single)
    list_muse_single=df_tmp.Code.tolist()
    
    # Read data
    df = pd.read_csv(in_csv, dtype={"MRID": str})
    df.columns = df.columns.astype(str)
       
    # Rename ROIs
    df_roidict = pd.read_csv(in_rois)
    df_roidict.Index = df_roidict.Index.astype(str)
    vdict = df_roidict.set_index('Index')['Code'].to_dict()
    df=df.rename(columns=vdict)
   
    # Keep DLICV in a separate df
    df_icv=df[['MRID', 'MUSE_702']]
    df_icv.columns=['MRID', 'DLICV']
    df=df.drop(columns=['MUSE_702'])
   
    # Add DLICV to demog
    df_demog = pd.read_csv(in_demog, dtype={"MRID": str})
    df_demog = df_demog.merge(df_icv, on='MRID')
    
    # Add covars
    df_raw=df.merge(df_demog, on=key_var)

    ###################################################
    # COMBAT
    
    # Select variables for harmonization
    muse_vars=df_raw.columns[df_raw.columns.str.contains('MUSE')].tolist()
    other_vars=['MRID', 'Age', 'Sex', 'SITE', 'DLICV']
    df_out=df_raw[other_vars + muse_vars]

    # Check that sample has age range consistent with the model
    df_out = df_out[df_out['Age'] > min_age]
    df_out = df_out[df_out['Age'] < max_age]

    # Save combat input
    f_combat_in=os.path.join(out_dir, f"{dset_name}_combat_in.csv")
    df_out.to_csv(f_combat_in, index=False)

    # Apply combat
    mdl_combat = os.path.join("src", "workflow", model_combat)
    f_combat_out=os.path.join(out_dir, f"{dset_name}_combat_out_init.csv")
    os.system(f"neuroharm -a apply -i {f_combat_in} -m {mdl_combat} -u {f_combat_out}")

    # Edit combat output (remove non mri columns and suffix combat)
    df_combat = pd.read_csv(f_combat_out, dtype={"MRID": str})
    df_combat.columns = df_combat.columns.astype(str)
    sel_vars = [key_var] + df_combat.columns[df_combat.columns.str.contains(suff_combat)].tolist()
    df_combat = df_combat[sel_vars]
    df_combat.columns = df_combat.columns.str.replace(suff_combat, "")

    # Add derived rois
    df_combat = combine_rois(df_combat, csv_muse_derived)

    # Merge covars to harmonized ROIs
    df_combat=df_demog.merge(df_combat, on=key_var)

    # Write out file
    f_combat_out=os.path.join(out_dir, f"{dset_name}_combat_out.csv")
    df_combat.to_csv(f_combat_out, index=False)

    ###################################################
    # Calculate Centiles
    
    # Normalize ROIs
    df_icvcorr=df_combat.copy()
    var_muse=df_icvcorr.columns[df_icvcorr.columns.str.contains('MUSE')]
    df_tmp=df_icvcorr[var_muse]
    df_tmp=df_tmp.div(df_combat['DLICV'], axis=0) * icv_ref_val
    #df_tmp=df_tmp.div(df_icvcorr['DLICV'], axis=0) * 100
    df_icvcorr[var_muse]=df_tmp.values

    # Calculate centiles
    df_cent=pd.read_csv(cent_csv)
    df_centiles=calc_subject_centiles(df_icvcorr, df_cent, df_roidict)

    # Write out files
    f_icvcorr=os.path.join(out_dir, f"{dset_name}_combat_icvcorr.csv")
    df_icvcorr.to_csv(f_icvcorr, index=False)
    f_centiles=os.path.join(out_dir, f"{dset_name}_combat_icvcorr_centiles.csv")
    df_centiles.to_csv(f_centiles, index=False)

    ###################################################
    # SPARE
    df_in=df_combat
    f_spare_in=os.path.join(out_dir, f"{dset_name}_spare_in.csv")
    df_in.to_csv(f_spare_in, index=False)

    # Apply spare
    bdir='/home/guraylab/GitHub/CBICA/NiChart_Project/src/workflow/models/vISTAG1/SPARE'
    pref='combined_DLMUSE_raw_COMBAT_SPARE-'
    suff='_Model.pkl.gz'
    df_spare=df_in[['MRID']]
    for spare_type in spare_types:        
        spare_mdl=os.path.join(bdir, f'{pref}{spare_type}{suff}')
        f_spare_out=os.path.join(out_dir, f'{dset_name}_SPARE_{spare_type}.csv')
        os.system(f"spare_score -a test -i {f_spare_in} -m {spare_mdl} -o {f_spare_out}")

        # Change column name for the spare output
        df = pd.read_csv(f_spare_out)
        df = df[df.columns[0:2]]
        df.columns = ['MRID', f'SPARE{spare_type}']
        df_spare=df_spare.merge(df)

    # Write out file with all spare scores
    f_spare_out=os.path.join(out_dir, f'{dset_name}_SPARE-ALL.csv')
    df_spare.to_csv(f_spare_out, index=False)

    ###################################################
    # SurrealGAN

    # Select input
    sel_covars=['MRID', 'Age', 'Sex', 'DLICV']
    sel_vars=sel_covars + list_muse_single
    df_in=df_raw[sel_vars]
    f_sgan_in=os.path.join(out_dir, f"{dset_name}_sgan_in.csv")
    df_in.to_csv(f_sgan_in, index=False)
    
    # Run prediction
    f_sgan_out=os.path.join(out_dir, f"{dset_name}_sgan_out.csv")
    cmd = f'PredCRD -i {f_sgan_in} -o {f_sgan_out}' 
    print(f'About to run {cmd}')
    os.system(cmd)
    
    ###################################################
    # Calculate Centiles
    
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

    # Run workflow
    print("Running: w_mlscores")
    run_workflow(options)

    print("Workflow complete! Output file:", options.dir_output)
