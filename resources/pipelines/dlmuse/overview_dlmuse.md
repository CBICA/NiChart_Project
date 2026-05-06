### DLMUSE: Automated Brain Anatomy Segmentation

Automatically segment T1 structural head MRI scans into 145 MUSE ROIs [1] using a fully automated pipeline that combines DLICV and DLMUSE. 

The pipeline first extracts the intra-cranial volume (ICV), and then segments it into regions of interest [2]. ROI volumes are computed both for individual ROIs derived directly from the segmentation mask, and for composite ROIs aggregated across multiple resolution levels.

#### Input

- T1-weighted scan (Nifti)

#### Output

- ICV mask (Nifti)
- ROI segmentation mask (Nifti)
- ROI volumes (csv file)

#### References

[1] Doshi J, et al. [MUSE: MUlti-atlas region Segmentation utilizing Ensembles of registration algorithms and parameters, and locally optimal atlas selection](https://pmc.ncbi.nlm.nih.gov/articles/PMC4806537). Neuroimage. 2016. 

[2] Bashyam VM et al. [DLMUSE: Robust Brain Segmentation in Seconds Using Deep Learning](https://pubs.rsna.org/doi/10.1148/ryai.240299). Radiol Artif Intell. 2025.

