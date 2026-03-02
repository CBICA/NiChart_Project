### DLWMLS: White Matter Hyperintensity Segmentation from FLAIR scans, divided into MUSE ROI levels

This pipeline segments White Matter Hyperintensities (WMH) from input FLAIR scans and seperates them into MUSE ROI level in seconds by combining DL-based methods [DLWMLS](https://github.com/CBICA/DLWMLS) and [DLMUSE](https://github.com/CBICA/NiChart_DLMUSE).


#### Input
- FLAIR + T1w scan pairs (Nifti)


#### Output
- Total WMH masks
- Regionally localized WMH masks (Nifti)
- Regionally localized WMH volumes (CSV)
