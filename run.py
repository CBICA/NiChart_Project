import argparse
import json
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_rois", help="Provide input csv name with ROIs", required=True)
    parser.add_argument("--input_demog", help="Provide input csv name with demographic info", required=True)
    parser.add_argument("--input_config", help="Provide input config file name", required=True)
    parser.add_argument("--dir_output", help="Provide output folder name", required=True)

    options = parser.parse_args()

    # Read default config file
    with open(options.input_config, 'r') as f:
        config = json.load(f)


    # Update the default config using the input args
    config['input_rois'] = options.input_rois
    config['input_demog'] = options.input_demog
    config['dir_output'] = options.dir_output

    # Create out dir
    if not os.path.exists(options.dir_output):
        os.makedirs(options.dir_output)    

    # Write config file
    json_config = json.dumps(config, indent=4)
    out_file = os.path.join(options.dir_output, 'config.json')
    with open("info.json", "w") as outfile:
        outfile.write(json_config)

    os.chdir('src/workflow/workflows/w_sMRI')
        os.system(f"snakemake --cores 1")
