# NiChart_Tissue_Segmentation

Brain tissue segmentation using FSL FAST and DLICV

## Overview

NiChart_Tissue_Segmentation offers easy brain tissue segmantation.

This is achieved through the [DLICV](https://github.com/CBICA/DLICV) and [FAST](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FAST) methods.

Given an input (sMRI) scan, NiChart_Tissue_Segmentation extracts the following:

1. Tissue segmentation
2. Volumetric data (optional)

This package uses [nnUNet](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1) (version 1) as a basis model architecture for the deep learning parts, [FAST](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FAST) for the tissue segmentation and various other [libraries](requirements.txt).

It is available both as an installable package, as well as a [docker container](https://hub.docker.com/repository/docker/aidinisg/nichart_tissue_segmentation/general). Please see [Installation](#installation) and [Usage](#usage) for more information on how to use it.

## Installation

1. Create a new conda env

    ```bash
    conda create --name NCTS python=3.8
    conda activate NCTS
    ```

    In one command:

    ```bash
    conda create --name NCTS -y python=3.8 && conda activate NCTS
    ```

2. Clone and install NiChart_Tissue_Segmentation

    ```bash
    git clone  https://github.com/CBICA/NiChart_Tissue_Segmentation.git
    cd NiChart_Tissue_Segmentation
    pip install .
    ```

3. Download model from this package's release as an [artifact](https://github.com/CBICA/NiChart_Tissue_Segmentation/releases/download/0.0.0/nnUNet_model.zip)

4. Run NiChart_Tissue_Segmentation. Please see [Usage](#usage) for an example.

## Docker-based build

The package comes already pre-built as a [docker container](https://hub.docker.com/repository/docker/aidinisg/nichart_tissue_segmentation/general), for convenience. Please see [Usage](#usage) for more information on how to use it. Alternatively, you can build the docker image locally using the dockerfile provided, like so:

```bash
docker build -t nichart_tissue_segmentation .
```

## Usage

Pre-trained nnUNet models for the skull-stripping task can be found in the [NiChart_Tissue_Segmentation - 0.0.0](https://github.com/CBICA/NiChart_Tissue_Segmentation/releases/tag/0.0.0) release as an [artifact](https://github.com/CBICA/NiChart_Tissue_Segmentation/releases/download/0.0.0/nnUNet_model.zip). Feel free to use it under the package's [license](LICENSE).

The model provided as an artifact is already in the file structure that's needed for the package to work, so make sure to include it as downloaded.

Given the following file structure:

```bash
temp
├── in                      // Input folder. Image names are irrelevant.
│   ├── image1.nii.gz
│   ├── image2.nii.gz
│   └── image3.nii.gz
├── nnUNet_model            // As provided from the release
│   └── nnUNet
└── out                     // Output destination
    ├── image1_seg.nii.gz
    ├── image2_seg.nii.gz
    ├── image3_seg.nii.gz
    └── output.csv
```

### As a locally installed package

A complete command would be (run from the directory of the package):

```bash
NiChart_Tissue_Segmentation -i /temp/in/ \
                            -o /temp/out/ \
                            -m /temp/nnUNet_model \
                            -c /temp/output.csv # Optional
```

For further explanation please refer to the complete documentation:

```bash
NiChart_Tissue_Segmentation -h
```

### Using the docker container

An example command using the [docker container](https://hub.docker.com/repository/docker/aidinisg/nichart_tissue_segmentation/general) is the following:

```bash
docker run -it --rm --gpus all -v ./:/workspace/ aidinisg/nichart_tissue_segmentation:0.0.0 NiChart_Tissue_Segmentation -i path/to/input -o path/to/output
```

Please note that the model is provided in the docker container, but you can always substitute it with your own nnUNet model.

### Example output

```bash
temp
├── in              
│   ├── image1.nii.gz
│   ├── image2.nii.gz
│   └── image3.nii.gz
├── nnUNet_model    
│   └── nnUNet
└── out                     // Output destination
    ├── image1_seg.nii.gz
    ├── image2_seg.nii.gz
    ├── image3_seg.nii.gz
    └── output.csv

```

## Contact

For more information, please contact [CBICA Software](mailto:software@cbica.upenn.edu).

## For Developers

Contributions are welcome! Please refer to our [CONTRIBUTING.md](CONTRIBUTING.md) for more information on how to report bugs, suggest enhancements, and contribute code.

If you're a developer looking to contribute, you'll first need to set up a development environment. After cloning the repository, you can install the development dependencies with:

```bash
pip install -r requirements-test.txt
```

This will install the packages required for running tests and formatting code. Please make sure to write tests for new code and run them before submitting a pull request.

© 2024 NiChart Team. All Rights Reserved.
