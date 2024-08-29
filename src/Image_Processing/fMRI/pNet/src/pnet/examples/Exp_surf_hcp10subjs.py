# This example is for applying pnet to surface fMRI data preprocessed with HCP pipelines
# 1. Specify the result folder directory in dir_pnet_result
# 2. Provide a txt formatted scan list file, such as Scan_List.txt
# 3. Use a prepared brain template file provided in pnet
# 4. Choose the desired number of FNs

# load pnet toolbox
import pnet

# Setup
# data type is Surface
dataType = 'Surface'
# data format is HCP surface, usually in CIFTI format, but can also be store as a 2D matrix in MAT file, fixed
dataFormat = 'HCP Surface (*.cifti, *.mat)'

# setup the output folder 
dir_pnet_result = '/mnt/8TBSSD/pnet_testing/FN17_Workflow_Surface'

# a txt file storing directory of each fMRI scan file, required to provide
file_scan = '/mnt/8TBSSD/pnet_testing/hcp_10_surfs.txt'

# a built-in brain template file, made for the HCP surface data (59412 vertices for cortical gray matter), optional to change
file_Brain_Template = pnet.Brain_Template.file_HCP_surf

# number of FNs, can be changed to any positive integer number
K = 17

# Setup number of scans loaded for each bootstrap run for estimating gFNs
sampleSize = 10 # a larger number is preferred for robustness, but should be no larger than the avaiable scans

# Setup number of runs for bootstraps
nBS = 5   # a larger number is preferred for robustness

# Setup number of time points for computing group FNs with bootstraps
nTPoints = 400  # a larger number is preferred for robustness

# Run pnet workflow
pnet.workflow_simple(
        dir_pnet_result=dir_pnet_result,
        dataType=dataType,
        dataFormat=dataFormat,
        file_scan=file_scan,
        file_Brain_Template=file_Brain_Template,
        K=K,
        sampleSize=sampleSize,
        nBS=nBS,
        nTPoints=nTPoints
    )
