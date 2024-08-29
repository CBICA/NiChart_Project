# Yuncong Ma, 2/13/2024
# A script to generate built-in brain template files
# Some data are missing due to data license, but can be downloaded online

import pnet
import os


# =============== HCP Surface =============== #
# Use shape file
dir_template = os.path.join(pnet.dir_brain_template, 'HCP')
file_surfL = os.path.join(dir_template, '32k_ConteAtlas_v2', 'Conte69.L.inflated.32k_fs_LR.surf.gii')
file_surfR = os.path.join(dir_template, '32k_ConteAtlas_v2', 'Conte69.R.inflated.32k_fs_LR.surf.gii')
file_maskL = os.path.join(dir_template, 'Gordon/Gordon2016Surface_parcellation_distribute-20agwt4', 'medial_wall.L.32k_fs_LR.func.gii')
file_maskR = os.path.join(dir_template, 'Gordon/Gordon2016Surface_parcellation_distribute-20agwt4', 'medial_wall.R.32k_fs_LR.func.gii')
file_surfL_inflated = os.path.join(dir_template, '32k_ConteAtlas_v2', 'Conte69.L.very_inflated.32k_fs_LR.surf.gii')
file_surfR_inflated = os.path.join(dir_template, '32k_ConteAtlas_v2', 'Conte69.L.very_inflated.32k_fs_LR.surf.gii')
#
if __name__ == '__main__':
    pnet.setup_brain_template(os.path.join(pnet.dir_brain_template, 'HCP_Surface'),
                          dataType='Surface', templateFormat='HCP',
                          file_surfL=file_surfL, file_surfR=file_surfR,
                          file_maskL=file_maskL, file_maskR=file_maskR,
                          maskValue=0,  # HCP uses 1 to label medial wall
                          file_surfL_inflated=file_surfL_inflated, file_surfR_inflated=file_surfR_inflated
                          )


# =============== HCP Surface-Volume =============== #
# Use a real HCP fMRI data to generate the volume parts
pnet.setup_cifti_volume(os.path.join(pnet.dir_example, 'HCP_Surface-Volume/Data/100206/rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii'),
                        os.path.join(pnet.dir_brain_template, 'HCP_Surface_Volume/CIFTI_Volume.nii.gz'))
# Use shape files to generate the surface part and combine the previously generated volume parts
if __name__ == '__main__':
    pnet.setup_brain_template(os.path.join(pnet.dir_brain_template, 'HCP_Surface_Volume'),
                          dataType='Surface-Volume', templateFormat='HCP',
                          file_surfL=file_surfL, file_surfR=file_surfR,
                          file_maskL=file_maskL, file_maskR=file_maskR,
                          maskValue=0,  # HCP uses 1 to label medial wall
                          file_surfL_inflated=file_surfL_inflated, file_surfR_inflated=file_surfR_inflated,
                          file_mask_vol=os.path.join(pnet.dir_brain_template, 'HCP_Surface_Volume/CIFTI_Volume.nii.gz'),
                          file_overlayImage=os.path.join(pnet.dir_brain_template, 'HCP_Surface_Volume/T1.nii.gz')
                          )

# =============== HCP Subcortex Volume =============== #
if __name__ == '__main__':
    pnet.setup_brain_template(os.path.join(pnet.dir_brain_template, 'HCP_Volume'),
                          dataType='Volume', templateFormat='HCP',
                          maskValue=1,
                          file_mask_vol=os.path.join(pnet.dir_brain_template, 'HCP_Surface_Volume/CIFTI_Volume.nii.gz'),
                          file_overlayImage=os.path.join(pnet.dir_brain_template, 'HCP_Surface_Volume/T1.nii.gz')
                          )

# =============== MNI Volume =============== #
dir_template = os.path.join(pnet.dir_brain_template, 'MNI_Volume')
file_mask_vol = os.path.join(dir_template, 'Brain_Mask.mat')
file_overlayImage = os.path.join(dir_template, 'T1.nii.gz')
if __name__ == '__main__':
    pnet.setup_brain_template(os.path.join(pnet.dir_brain_template, 'MNI_Volume'),
                          dataType='Volume', templateFormat='3D Matrix',
                          file_mask_vol=file_mask_vol,
                          maskValue=1,
                          file_overlayImage=file_overlayImage
                          )

# =============== FreeSurfer Surface =============== #
dir_template = os.path.join(pnet.dir_brain_template, 'FS')
file_surfL = os.path.join(dir_template, 'fsaverage5/surf/lh.pial')
file_surfR = os.path.join(dir_template, 'fsaverage5/surf/rh.pial')
file_maskL = os.path.join(dir_template, 'mask_files/lh.Mask_SNR.label')
file_maskR = os.path.join(dir_template, 'mask_files/rh.Mask_SNR.label')
file_surfL_inflated = os.path.join(dir_template, 'fsaverage5/surf/lh.inflated')
file_surfR_inflated = os.path.join(dir_template, 'fsaverage5/surf/rh.inflated')
if __name__ == '__main__':
    pnet.setup_brain_template(os.path.join(pnet.dir_brain_template, 'FreeSurfer_fsaverage5'),
                          dataType='Surface', templateFormat='FreeSurfer',
                          file_surfL=file_surfL, file_surfR=file_surfR,
                          file_maskL=file_maskL, file_maskR=file_maskR,
                          maskValue=0,  # mask file uses 1 to label unused voxels
                          file_surfL_inflated=file_surfL_inflated, file_surfR_inflated=file_surfR_inflated
                          )




