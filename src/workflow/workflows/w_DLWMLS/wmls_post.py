import argparse
import os
import glob
import nibabel as nib
import pandas as pd
import numpy as np

def wmls_post(in_dir, img_suff, out_csv):
    pattern = os.path.join(in_dir, f"*{img_suff}")
    sel_files = glob.glob(pattern, recursive=False)
    df = pd.DataFrame(columns = ['MRID', 'WMLVol'])
    for tmp_sel in sel_files:
        nii = nib.load(tmp_sel)
        data = nii.get_fdata().flatten()
        wmlvol = np.prod(nii.header.get_zooms()) * sum(data>0)
        mrid = os.path.basename(tmp_sel).replace(img_suff, '')
        df.loc[len(df)] = {'MRID':mrid, 'WMLVol':wmlvol}
    if df.shape[0] > 0:
        df.to_csv(out_csv, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_dir", help="Provide the path to data", required=True
    )
    parser.add_argument(
        "--in_suff", help="Provide the image suffix", required=True
    )
    parser.add_argument(
        "--out_csv", help="Provide the out csv name", required=True
    )
    options = parser.parse_args()
    
    wmls_post(options.in_dir, options.in_suff, options.out_csv)
    
