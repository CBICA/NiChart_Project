## input:
- T1 scan (one or multiple, required)

## output:
- Segmentation mask (one for each scan)
- ROI volumes (single csv file with ROI volumes for all scans)

   
## example:
📁 my_project
│
📥 Input
├── 📁 t1
│   ├── 📄 scan1_T1.nii.gz
│   └── 📄 scan2_T1.nii.gz
│
📤 Output
├── 📁  DLMUSE_seg
│   ├── 📄 scan1_T1_DLMUSE.nii.gz
│   └── 📄 scan2_T1_DLMUSE.nii.gz
└── 📁 DLMUSE_vol
    └── 📄 DLMUSE_Volumes.csv
```

