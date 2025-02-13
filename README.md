# NiChart: Neuro-imaging Chart

NiChart is a comprehensive framework designed to revolutionize neuroimaging research. It offers large-scale neuroimaging capabilities, sophisticated analysis methods, and user-friendly tools, all seamlessly integrated into a local installation version and the [AWS Cloud](https://neuroimagingchart.com/portal/).

## Components

1. **Image Processing**: Utilizes tools like [DLMUSE](https://github.com/CBICA/NiChart_DLMUSE), [fMRIPrep](https://github.com/nipreps/fmriprep) [XCEPengine](https://github.com/PennLINC/xcp_d), and [QSIPrep](https://github.com/PennLINC/qsiprep) for effective image analytics.
2. **Reference Data Curation**: Houses ISTAGING, 70000 Scans, and 14 individual studies to provide curated reference data.
3. **Data Harmonization**: Employs [neuroharmonize](https://github.com/rpomponio/neuroHarmonize) and [Combat](https://github.com/Zheng206/ComBatFam_Pipeline) for ensuring consistent data standards.
4. **Machine Learning Models**: Provides Supervised, Semi-supervised, and DL Models for advanced neuroimaging analysis including [SpareScore](https://github.com/CBICA/spare_score).
5. **Data Visualization**: Features like Centile curves, direct image linking, and reference values for comprehensive data visualization.
6. **Deployment**: Supports open-source Github components and Docker container compatibility deployed in a local environment & [AWS Cloud](https://aws.amazon.com/).


## System Requirements

For recommended system configuration, please refer to: [nnUNet hardware requirements](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md#hardware-requirements).

## Installation Instructions
  ```bash
  git clone https://github.com/CBICA/NiChart_Project.git
  pip install -r requirements.txt
  ```

## Run NiChart Locally (GUI)
```bash
cd src/viewer/
streamlit run NiChartProject.py
```
The app will start in your localhost.

## Quick Links

[![NiChart Website & Cloud](https://img.shields.io/badge/-Website-blue?style=for-the-badge&logo=world&logoColor=white)](https://neuroimagingchart.com/) [![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)](https://hub.docker.com/u/cbica) [![AIBIL Research](https://img.shields.io/badge/-Research-blue?style=for-the-badge&logo=google-scholar&logoColor=white)](https://aibil.med.upenn.edu/research/) [![YouTube](https://img.shields.io/badge/YouTube-%23FF0000.svg?style=for-the-badge&logo=YouTube&logoColor=white)](https://www.youtube.com/@NiChart-UPenn)

[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/NiChart_AIBIL.svg?style=social&label=Follow%20%40NiChart_AIBIL)](https://x.com/NiChart_AIBIL)

© 2024 CBICA. All Rights Reserved.

test
