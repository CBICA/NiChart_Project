### DLMUSE: Robust Brain Segmentation in Seconds using Deep Learning

A complete brain parcellation pipeline from T1 structural head MRI images into 145 [MUSE](https://pmc.ncbi.nlm.nih.gov/articles/PMC4806537/) ROIs.

The pipeline offers easy ICV (Intra-Cranial Volume) mask extraction, and brain segmentation into ROIs by combining DLICV and [DLMUSE](https://pubmed.ncbi.nlm.nih.gov/40960397/) methods. Intermediate step results are saved for easy access to the user.

Given an input MRI scan (T1 structural), NiChart_DLMUSE extracts the following:

        1. ICV mask
        2. Brain MUSE ROI segmentation
        3. ROI volumes in a .csv format
        4. Individual ROI mask (optionally).

#### Input

- T1-weighted scan (Nifti)


#### Output

- Segmentation labels (Nifti)
- Volumes of ROIs (csv file)


#### Reference

Bashyam VM et al., Alzheimer’s Disease Neuroimaging Initiative; iSTAGING Consortium. DLMUSE: Robust Brain Segmentation in Seconds Using Deep Learning. Radiol Artif Intell. 2025 Nov;7(6):e240299. doi: 10.1148/ryai.240299. PMID: 40960397.