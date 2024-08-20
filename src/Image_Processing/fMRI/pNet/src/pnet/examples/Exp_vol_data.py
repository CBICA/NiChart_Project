# This example is for applying pnet to volumtric fMRI data spatially normalized to MNI space
# 1. Specify the result folder directory in dir_pnet_result
# 2. Provide a txt formatted scan list file, such as Scan_List.txt
# 3. Use a prepared brain template file provided in pnet
# 4. Choose the desired number of FNs

# load pnet toolbox
import pnet

# Setup
# data type is volume
dataType = 'Volume'

 # data format is NIFTI, which stores a 4D matrix, fixed for this example
dataFormat = 'Volume (*.nii, *.nii.gz, *.mat)'

# setup the output folder 
dir_pnet_result = '/mnt/8TBSSD/pnet_testing/FN17_Workflow_Volume'

# a txt file storing directory of each fMRI scan file, required to provide
file_scan = '/mnt/8TBSSD/pnet_testing/Sub_testlist.txt'

# a built-in brain template file, MNI standard space (2mm isotropic with 91*109*91 voxels)
file_Brain_Template = pnet.Brain_Template.file_MNI_vol

# number of FNs, can be changed to any positive integer number
K = 7

# Setup number of scans loaded for each bootstrap run for estimating gFNs
sampleSize = 10 # a larger number is preferred for robustness, but should be no larger than the avaiable scans

# Setup number of runs for bootstraps
nBS = 5   # a larger number is preferred for robustness

# Setup number of time points for computing group FNs with bootstraps
nTPoints = 200  # a larger number is preferred for robustness

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
