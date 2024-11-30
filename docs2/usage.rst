#####
Usage
#####

Pre-trained nnUNet models for the skull-stripping can be found in `HuggingFace nichart/DLICV <https://huggingface.co/nichart/DLICV/tree/main>`_ and
segmentation tasks can be found in `HuggingFace nichart/DLMUSE <https://huggingface.co/nichart/DLMUSE/tree/main>`_. Feel free to use it under the package's license.

*********
Using CLI
*********

Our CLI is simply to use as it just takes 2 required parameters: ::

    $ NiChart_DLMUSE -i <input path> -o <output path>

By default, our CLI runs on CPU. You can run it on GPU with CUDA or MPS just by specifying the device: ::

    $ NiChart_DLMUSE -i <input path> -o <output path> -d cuda


.. note::

    **MPS** Can be used if your chip supports 3d convolution(M1 chips do not support 3d convolution, thus, you can't use MPS with M1 macbooks)


We use batch splitting and paralellization to accelerate the procedure. By default, we split the input data into
4 subfolders. You can specify the number of subfolders with the ``-c`` or ``--cores`` option: ::

    $ NiChart_DLMUSE ... -c 6

This will create 6 subfolders instead of the default 4.

We also support ``BIDS`` I/O in our latest stable release. In order to run NiChart DLMUSE with a BIDS folder as the input you
need to have one T1 image under the anat subfolder. After the run NiChart DLMUSE will return the segmented images in the same
subfolders. If you have a `BIDS` input folder you have to specify it at the CLI command: ::

    $ NiChart_DLMUSE ... --bids 1

For further explanation please refer to the complete CLI documentation: ::

    $ NiChart_DLMUSE -h


**************************
Using the docker container
**************************

Using the file structure explained above, an example command using the `docker container <https://hub.docker.com/r/cbica/nichart_dlmuse/tags>`_
is the following: ::

    # For GPU
    $ CUDA_VERSION=11.8 docker pull cbica/nichart_dlmuse:1.0.1-cuda${CUDA_VERSION}

    # For CPU
    docker pull cbica/nichart_dlmuse:1.0.4-default

    $ # Run the container with proper mounts, GPU enabled
      # Place input in /path/to/input/on/host.
      # Replace "-d cuda" with "-d mps" or "-d cpu" as needed...
      # or don't pass at all to automatically use CPU.
      # Each "/path/to/.../on/host" is a placeholder, use your actual paths!
      docker run -it --name DLMUSE_inference --rm
          --mount type=bind,source=/path/to/input/on/host,target=/input,readonly
          --mount type=bind,source=/path/to/output/on/host,target=/output
          --gpus all cbica/nichart_dlmuse -d cuda


*******************************
Using the singularity container
*******************************

We do not recommend using the singularity container as is currently outdated and not maintaned: ::

    $ singularity run --nv --containall --bind /path/to/.\:/workspace/ nichart_dlmuse.simg NiChart_DLMUSE -i /workspace/temp/nnUNet_raw_data_base/nnUNet_raw_data/ -o /workspace/temp/nnUNet_out -p structural --derived_ROI_mappings_file /NiChart_DLMUSE/shared/dicts/MUSE_mapping_derived_rois.csv --MUSE_ROI_mappings_file /NiChart_DLMUSE/shared/dicts/MUSE_mapping_consecutive_indices.csv --nnUNet_raw_data_base /workspace/temp/nnUNet_raw_data_base/ --nnUNet_preprocessed /workspace/temp/nnUNet_preprocessed/ --model_folder /workspace/temp/nnUNet_model/ --all_in_gpu True --mode fastest --disable_tta
