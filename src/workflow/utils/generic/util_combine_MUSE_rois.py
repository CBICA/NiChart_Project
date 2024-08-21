import csv as csv
import nibabel as nib
import numpy as np
import pandas as pd
import sys

def combine_rois(in_csv, dict_csv, out_csv, key_var = 'MRID', roi_prefix = 'MUSE_'):
    '''
    Calculates a dataframe with the volumes of derived + single rois.
    - Derived ROIs are calculated by adding composing single ROIs, read from the dictionary
    - If a derived ROI already exists in input data, it will not be updated
    '''

    # Read input files
    df_in = pd.read_csv(in_csv, dtype = {'MRID':str})
    
    ## Read derived roi map file to a dictionary
    dict_roi = {}
    with open(dict_csv) as roi_map:
        reader = csv.reader(roi_map, delimiter=',')
        for row in reader:
            key = roi_prefix + str(row[0])
            val = [roi_prefix + str(x) for x in row[2:]]
            dict_roi[key] = val

    ## Create derived roi df
    label_names = list(dict_roi.keys())
    df_out = df_in[[key_var]].copy()
    df_out = df_out.reindex(columns = [key_var] + label_names)

    # Calculate volumes for derived rois
    for i, key in enumerate(dict_roi):
        key_vals = dict_roi[key]
        try:
            if key in df_in.columns:
                df_out[key] = df_in[key]                    # If ROI is already in df, do not calculate it from parts
                print('WARNING:  Skip derived ROI, already in data: ' + key) 
            else:
                df_out[key] = df_in[key_vals].sum(axis=1)   # If not, calculate it (sum of single ROIs in the dict)
        except:
            df_out = df_out.drop(columns = key)
            print('WARNING:  Skip derived ROI with missing components: ' + key) 
        
    ## Save df_out
    df_out.to_csv(out_csv, index = False)


if __name__ == "__main__":
    # Access arguments from command line using sys.argv
    if len(sys.argv) != 6:
        print("Error: Please provide all required arguments")
        print("Usage: python combine_MUSE_rois.py in_csv.csv dict_csv.csv key_var roi_prefix out_csv.csv")
        sys.exit(1)

    in_csv = sys.argv[1]
    dict_csv = sys.argv[2]
    key_var = sys.argv[3]
    roi_prefix = sys.argv[4]
    out_csv = sys.argv[5]
    
    # Print run command
    print('About to run: ' + ' '.join(sys.argv))    

    # Call the function
    combine_rois(in_csv, dict_csv, out_csv, key_var, roi_prefix)

    print("Derived roi calculation complete! Output file:", out_csv)
