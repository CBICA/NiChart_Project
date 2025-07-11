### BAScores: Direct Image-to-Biomarker Brain Age Scores

#### Input:

- T1 scan (one or multiple, required)

#### Output:

- Inference results (single CSV file with predicted brain age scores for all scans)

- Attention maps (one NIFTI file for each input scan)

#### Example:
```
📁 my_project
│
📥 Input
├── 📁 t1
│   ├── 📄 scan1_T1.nii.gz
│   └── 📄 scan2_T2.nii.gz
│
📤 Output
├── 📁  bascores_attentionmaps
│   ├── 📄 scan1_T1_attentionmap.nii.gz
│   └── 📄 scan2_T1_attentionmap.nii.gz
└── 📁 bascores_csv
    └── 📄 my_project_dlmuse.csv
```