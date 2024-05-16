import argparse
import pandas as pd
from spare_scores import spare_test

def merge_and_test(roi_file, 
                   demographic_file, 
                   model_path, 
                   key_var,
                   output_file,
                   spare_column_name,
                   verbose,
                   logs_file):
    
    # Read the input CSV files into pandas DataFrames
    df_roi = pd.read_csv(roi_file)
    df_demographic = pd.read_csv(demographic_file)

    # Merge the DataFrames based on the desired column
    if not key_var:
        key_var = df_roi.columns[0]
    merged_df = pd.merge(df_roi, df_demographic, on=key_var)

    # Call the spare_test function from the spare_scores package
    result = spare_test(merged_df, 
                        model_path,
                        key_var,
                        output_file,
                        spare_column_name,
                        verbose,
                        logs_file)

    return result

if __name__ == "__main__":
    # Example usage:
    # python merge_ROI_demo_and_test.py -i spare_scores/data/example_data_ROIs.csv \
    #                                   -d spare_scores/data/example_data_demographics.csv \
    #                                   -m spare_scores/mdl/mdl_SPARE_AD_hMUSE_single.pkl.gz \
    #                                   -kv ID \
    #                                   -o zzz_output.csv \
    #                                   -l zzz_logs.txt \
    #                                   -sv SPARE_score
    # # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Spare Scores Analysis')

    # Define the command-line arguments
    parser.add_argument('-i', '--input', required=True, help='Input ROI CSV file')
    parser.add_argument('-d', '--demographic', required=True, help='Input demographic CSV file')
    parser.add_argument('-m', '--model', required=True, help='Model for spare_train')
    parser.add_argument('-kv', '--key_var', required=False, default='', help='The key variable of the dataset.')
    parser.add_argument('-o', '--output', required=False, default='', help='Output CSV file')
    parser.add_argument('-l', '--logs', required=False, default='', help='Output logs file')
    parser.add_argument('-sv', '--spare_var', required=False, default='SPARE_score', help='Column for calculated spare score')
    parser.add_argument('-v', '--verbose', required=False, default=1, help='Type of logging messages.')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the merge_and_test function with the provided arguments
    merge_and_test(args.input, 
                   args.demographic, 
                   args.model, 
                   args.key_var,
                   args.output, 
                   args.spare_var,
                   args.verbose, 
                   args.logs)
