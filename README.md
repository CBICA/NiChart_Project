# NiChart: Neuro Imaging Chart

NiChart is a comprehensive framework designed to revolutionize neuroimaging research. It offers large-scale neuroimaging capabilities, sophisticated analysis methods, and user-friendly tools, all integrated seamlessly into the AWS Cloud.

## Components

1. **Image Processing**: Utilizes tools like [DLMUSE](https://github.com/CBICA/niCHARTPipelines), [fMRIPrep](https://github.com/nipreps/fmriprep) [XCEPengine](https://github.com/PennLINC/xcp_d), and [QSIPrep](https://github.com/PennLINC/qsiprep) for effective image analytics.
2. **Reference Data Curation**: Houses ISTAGING, 70000 Scans, and 14 individual studies to provide curated reference data.
3. **Data Harmonization**: Employs [neuroharmonize](https://github.com/rpomponio/neuroHarmonize) and [Combat](https://github.com/Zheng206/ComBatFam_Pipeline) for ensuring consistent data standards.
4. **Machine Learning Models**: Provides Supervised, Semi-supervised, and DL Models for advanced neuroimaging analysis.
5. **Data Visualization**: Features like Centile curves, direct image linking, and reference values for comprehensive data visualization.
6. **Deployment**: AWS Cloud App support with open-source Github components and Docker container compatibility.

## Quick Links

[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)](https://hub.docker.com/u/cbica)

[![NiChart Website](https://img.shields.io/badge/-Website-blue?style=for-the-badge&logo=world&logoColor=white)](https://neuroimagingchart.com/)

[![AIBIL Research](https://img.shields.io/badge/-Research-blue?style=for-the-badge&logo=google-scholar&logoColor=white)](https://aibil.med.upenn.edu/research/)

[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/NiChart_AIBIL.svg?style=social&label=Follow%20%40NiChart_AIBIL)](https://x.com/NiChart_AIBIL)

[![YouTube](https://img.shields.io/badge/YouTube-%23FF0000.svg?style=for-the-badge&logo=YouTube&logoColor=white)](https://www.youtube.com/@NiChart-UPenn)


## Installation Instructions

1. `git clone https://github.com/CBICA/NiChart_Project.git`
2. `git submodule update --init --recursive --remote`
3. `pip install -r requirements.txt`
4. (If needed for your system/CUDA version) Follow the [PyTorch installation instructions.](https://pytorch.org/get-started/locally/). Unless you receive CUDA errors while running NiChart, this is probably not needed.

## Example Usage

### Run Locally
```bash
cd src/NiChart_Viewer/src
streamlit run NiChartProject.py
```

The app will start, running on localhost, port 8501.
Your browser should generally open automatically if you are running locally. 
While it is running, you can connect by opening your browser and navigating to http://localhost:8501

When you close your terminal, the server will stop running.

### Run with Docker

First, make sure you have [installed Docker.](https://docs.docker.com/engine/install/)

Some example commands for the [docker container](https://hub.docker.com/repository/docker/cbica/nichart/general) are the following:


```bash
# Pull the image for your CUDA version (as needed)
CUDA_VERSION=11.8 docker pull cbica/nichart:1.0.0-cuda${CUDA_VERSION}
# or, for CPU:
docker pull cbica/nichart:1.0.0

## Suggested automatic inference run time command 
## Place input in /path/to/input/on/host.
## Each "/path/to/.../on/host" is a placeholder, use your actual paths!
docker run -it --name nichart_server --rm -p 8501:8501
    --mount type=bind,source=/path/to/input/on/host,target=/input,readonly 
    --mount type=bind,source=/path/to/output/on/host,target=/output
    --gpus all cbica/nichart:1.0.0
## Run the above, then open your browser to http://localhost:8501 
## The above runs the server in your terminal, use Ctrl-C to end it.
## To run the server in the background, replace the "-it" flag in the command with "-d".
## To end the background server, use "docker stop nichart_server"
## DO NOT USE this as a public web server!
```

For a full explanation of Docker options, see the [docker run documentation.](https://docs.docker.com/reference/cli/docker/container/run/)

### Run with Singularity/Apptainer
Coming soon!
---

© 2024 CBICA. All Rights Reserved.
