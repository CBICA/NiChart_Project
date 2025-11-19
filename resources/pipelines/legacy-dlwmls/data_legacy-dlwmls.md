### DLWMLS: Automatic Segmentation of White Matter Lesions

#### Input:

- FL scan (one or multiple, required)
- T1 scan (one or multiple, optional) [COMING SOON]

#### Output:

- Lesion segmentation mask (one for each scan)

- Lesion segmentation mask in T1 space (one for each scan; if T1 image is available) [COMING SOON]

- Regional lesion volumes (single csv file; if T1 image is available) [COMING SOON]

#### Example:
```
Only FL images:
📁 my_project
│
📥 Input
├── 📁 fl
│   ├── 📄 scan1_FL.nii.gz
│   └── 📄 scan2_FL.nii.gz
│
📤 Output
└── 📁  DLWMLS_seg
    ├── 📄 scan1_FL_DLWMLS.nii.gz
    └── 📄 scan2_FL_DLWMLS.nii.gz

FL + T1 images [COMING SOON]:
📁 my_project
│
📥 Input
├── 📁 fl
│   ├── 📄 scan1_FL.nii.gz
│   └── 📄 scan2_FL.nii.gz
├── 📁 t1
│   ├── 📄 scan1_T1.nii.gz
│   └── 📄 scan2_T1.nii.gz
│
📤 Output
└── 📁  DLWMLS_seg
│   ├── 📄 scan1_FL_dlwmls.nii.gz
│   ├── 📄 scan2_FL_dlwmls.nii.gz
│   ├── 📄 scan1_FL_dlwmls_inT1.nii.gz
│   └── 📄 scan2_FL_dlwmls_inT1.nii.gz
└── 📁  DLWMLS_vol
    └── 📄 my_project_dlwmls.csv

```

