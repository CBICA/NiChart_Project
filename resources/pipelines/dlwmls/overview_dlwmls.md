### DLMUSE-DLWMLS: Regional + Lesion Segmentation from T1 + FLAIR Brain Scans 
This pipeline combines DLMUSE and DLWMLS. T1 and FLAIR images are aligned and the T1 is segmented via DLMUSE. White matter lesions in FLAIR are segmented via DLWMLS. Both segmentations are used to calculate regionally localized lesion volumes.

#### Source:

https://github.com/CBICA/NiChart_DLMUSE
https://github.com/CBICA/NiChart_DLWMLS

#### Input

- T1-weighted scans (Nifti)
- FL-weighted scans (Nifti)

#### Output

-  Regionally localized lesion volumes (CSV)
