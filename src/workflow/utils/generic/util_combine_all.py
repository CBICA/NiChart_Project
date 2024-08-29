import pandas as pd
import sys
import os

def combine_all(out_csv, list_in_csv):
    """
    Combines final output files
    """
    
    key_var = 'MRID'
    
    df_demog = pd.read_csv(list_in_csv[0])

    df_roi = pd.read_csv(list_in_csv[1])
    df_roi = df_roi[df_roi.Name != 'ICV']
    
    df_data = pd.read_csv(list_in_csv[2])
    df_norm = pd.read_csv(list_in_csv[3])
    df_harm = pd.read_csv(list_in_csv[4])
    df_spare = pd.read_csv(list_in_csv[5])

    df_icv = df_data[['MRID','MUSE_702']]
    df_icv.columns = ['MRID','ICV']

    #print([['MRID'] +  df_roi.Code.to_list()])
    #input()

    df_data = df_data.rename(columns=dict(zip(df_roi.Code, df_roi.Name)))
    df_data = df_data[['MRID'] +  df_roi.Name.to_list()]

    df_norm = df_norm.rename(columns=dict(zip(df_roi.Code, df_roi.Name)))
    df_norm = df_norm[['MRID'] +  df_roi.Name.to_list()]
    
    df_harm = df_harm.rename(columns=dict(zip(df_roi.Code, df_roi.Name)))
    df_harm = df_harm[['MRID'] +  df_roi.Name.to_list()]
    
    df_out = df_icv.merge(df_data, on='MRID')    
    df_out = df_data.merge(df_norm, on='MRID', suffixes=['','_normICV'])
    df_out = df_out.merge(df_harm, on='MRID', suffixes=['','_COMBAT'])
    df_out = df_out.merge(df_spare, on='MRID')
    
    df_out = df_demog.merge(df_out, on='MRID')

    # Write out file
    df_out.to_csv(out_csv, index=False)
        
if __name__ == "__main__":
  # Access arguments from command line using sys.argv
  if len(sys.argv) < 3:
      print("Error: Please provide all required arguments")
      print("Usage: python combine_data.py out_csv.csv in_csv1.csv,in_csv2.csv,...")
      sys.exit(1)

  out_csv = sys.argv[1]
  list_in_csv = sys.argv[2:]
  
  # Call the function
  combine_all(out_csv, list_in_csv)

  print("Merge complete! Output file:", out_csv)
