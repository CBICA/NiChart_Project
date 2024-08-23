# load pnet toolbox
import pnet

import argparse

import tomli
def read_config(file_path):
  try:
    with open(file_path, "rb") as f:
        config = tomli.load(f)  # , 'owner')
        necessary_settings = config['necessary_settings']
        pFN_settings = config['pFN_settings']
        gFN_settings = config['gFN_settings']
        hpc_settings = config['hpc_settings']
    return config #necessary_settings, pFN_settings, gFN_settings, hpc_settings
  except tomli.TOMLDecodeError:
    print(f"errors in {file_path}.")

def main(config_file='config.txt', HPC=None):
    # get configuration information
    config = read_config(config_file)
    # necessary_settings
    dataType = config['necessary_settings']['dataType']
    dataFormat = config['necessary_settings']['dataFormat']
    dir_pnet_result = config['necessary_settings']['dir_pnet_result']
    file_scans = config['necessary_settings']['file_scans']
    file_Brain_Template = config['necessary_settings']['file_Brain_Template']
    K = config['necessary_settings']['K']
    method = config['necessary_settings']['method']
    #pFN_settings
    if config['pFN_settings']['file_gFN']  == "None":
        file_gFN = None
    else:
        file_gFN = config['pFN_settings']['file_gFN']
    #gFN_setting
    sampleSize = config['gFN_settings']['sampleSize']
    nBS = config['gFN_settings']['nBS']
    nTPoints = config['gFN_settings']['nTPoints']
    #hpc_settings
    pnet_env = config['hpc_settings']['pnet_env']
    hpc_submit = config['hpc_settings']['submit']
    hpc_computation_resource = config['hpc_settings']['computation_resource']
    #print(f"computation resource: {hpc_computation_resource}")

    print(f"dataType: {dataType}")
    print(f"dataFormat: {dataFormat}")
    print(f"dir_pent_result: {dir_pnet_result}")
    print(f"file_scan: {file_scans}")
    print(f"file_Brain_Template: {file_Brain_Template}")
    print(f"K: {K}")
    print(f"sampleSize: {sampleSize}")
    print(f"nBS: {nBS}")
    print(f"nTPoints: {nTPoints}")

    if HPC is None:
       pnet.workflow(
           dir_pnet_result=dir_pnet_result,
           dataType=dataType,
           dataFormat=dataFormat,
           file_scan=file_scans,
           file_Brain_Template=file_Brain_Template,
           K=K,
           method=method,
           init='random', 
           sampleSize=sampleSize,
           nBS=nBS,
           nTPoints=nTPoints,
           Computation_Mode='CPU_Torch',
           Combine_Scan=False)

    elif HPC == 'qsub':
       print(f"HCP={HPC}")
       pnet.workflow_cluster(
           dir_pnet_result=dir_pnet_result,
           dataType=dataType,
           dataFormat=dataFormat,
           file_Brain_Template=file_Brain_Template,
           file_scan=file_scans,
           file_subject_ID=None,
           file_subject_folder=None,
           K=K,
           Combine_Scan=False,
           sampleSize=sampleSize,
           nBS=nBS,
           nTPoints=nTPoints,
           dir_env=pnet_env['dir_env'],
           dir_python=pnet_env['dir_python'],
           dir_pnet=pnet_env['dir_pnet'],
           Computation_Mode='CPU_Torch',
           submit_command=hpc_submit['submit_command'],
           thread_command=hpc_submit['thread_command'],
           memory_command=hpc_submit['memory_command'],
           log_command=hpc_submit['log_command'],
           computation_resource=hpc_computation_resource
       )
    else:
       print(f"Error: {HPC} is not supported yet!")

if __name__ == "__main__":
    #Create the parser
    parser = argparse.ArgumentParser(description="pNet: a toolbox for computing personalized functional networks from preprocessed functional magnetic resonance imaging (fMRI) data")
    # Add arguments
    parser.add_argument("-c", "--config", type=str, help="A configuration file for setting parameters", required=True)
    parser.add_argument("--hpc", type=str, default=None, help="HPC computing: None (default:not available) or qsub", required=False)
    # Parse the arguments
    args = parser.parse_args()
    config_file = args.config
    hpc = args.hpc
    main(config_file, hpc)

