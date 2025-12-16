### DLWMLS: Automatic Segmentation of White Matter Lesions

#### Source:

https://github.com/CBICA/DLWMLS

#### Description

DLWMLS uses a trained nnUNet model to compute the segmentation of white matter lesions

#### Input

- FL-weighted scan (Nifti, Required)
- T1-weighted scan (Nifti, Optional)


#### Output

- Segmentation labels (Nifti)
- Lesion volumes in ROIs (csv, if T1 scan is provided)

