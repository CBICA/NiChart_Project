# pNet
A Toolbox for Personalized Functional Network Modeling

> This toolbox is to use pre-computed group-level functional networks (gFNs) to obtain personalized function networks (pFNs) for new fMRI data
> It supports volume based data (.nii or .nii.gz) in MNI space (matrix size = 91x109x91)

Code Example
> /Script/Extract_MNI_Volume_pFN.sh -k 17 -exp Test -fmri /Data/Test/Example_fMRI.nii.gz

Result File
> Results will be stored in a subfolder in /Result, named by the experiment name
> finual_UV.mat store the pFNs (V) and their corresponding timecourses (U)
