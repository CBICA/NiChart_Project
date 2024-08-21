import pandas as pd
import argparse
import sys

def corr_icv(in_csv, corr_type, icv_var, exclude_vars, suffix, out_csv):
    """
    Calculates ICV corrected values
    """
    
    # Read input file
    df = pd.read_csv(in_csv)
    
    # Get var groups
    list_exclude = exclude_vars.split(',') + [icv_var]
    list_target = df.columns[df.columns.isin(list_exclude) == False]
    df_p1 = df[list_exclude]
    df_p2 = df[list_target]
    val_icv = df[icv_var]

    # Set correction factor
    if corr_type == 'percICV':
        corr_val = 100
    if corr_type == 'normICV':
        corr_val = 1430000      ## Average ICV
    
    # Add suffix to corrected variables
    if suffix != 'NONE':
        df_p2 = df_p2.add_suffix(suffix)
        
    # Correct ICV
    df_p2 = df_p2.div(val_icv, axis=0)*corr_val
    
    # Combine vars
    df_out = pd.concat([df_p1, df_p2], axis = 1)
    
    # Write out file
    df_out.to_csv(out_csv, index=False)

if __name__ == "__main__":
    # Access arguments from command line using sys.argv
    if len(sys.argv) != 7:
        print("Error: Please provide all required arguments")
        print("Usage: python select_sample.py in_csv.csv in_filter.json out_csv.csv")
        sys.exit(1)

    in_csv = sys.argv[1]
    corr_type = sys.argv[2]
    icv_var = sys.argv[3]
    exclude_vars = sys.argv[4]
    suffix = sys.argv[5]
    out_csv = sys.argv[6]

    # Print run command
    print('About to run: ' + ' '.join(sys.argv))

    # Call the function
    # Call the function
    corr_icv(in_csv, corr_type, icv_var, exclude_vars, suffix, out_csv)

    print("ICV correction complete! Output file:", out_csv)

