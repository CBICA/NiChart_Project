import argparse
import os
import glob
import nibabel as nib

def wmls_post(in_dir, img_suff):
    pattern = os.path.join(in_dir, f"*{img_suff}")
    sel_files = glob.glob(pattern, recursive=False)
    df = pd.DataFrame(columns = ['MRID', 'WMLVol'])
    for tmp_sel in sel_files:
        nii = nib.load(tmp_sel)
        data = nii.get_fdata().flatten()
        wmlvol = np.prod(nii.get_zooms) * sum(data>0)
        mrid = os.path.basename(file_path).str.replace(img_suff, '')
        df.loc[len(df)] = {'MRID':mrid, 'WMLVol':wmlvol}
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i, --indir", help="Provide the path to data", required=True
    )
    options = parser.parse_args()

    wmls_post(indir)
    
