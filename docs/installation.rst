############
Installation
############

Users can install **NiChart Project** with pip: ::

    pip install NiChart_Project

Alternatively, the package can be installed from source: ::

    git clone https://github.com/CBICA/NiChart_Project
    cd NiChart_Project && python3 -m pip install -e .

We release our latest stable version on PyPI; accordingly, **we strongly recommend pip installation**.

.. warning::
    PyTorch and NumPy have known compatibility issues across different platforms. To avoid potential conflicts, please follow the installation instructions below.

- After installing all other necessary packages, uninstall any existing Torch installations: ::

   $ pip uninstall torch

- Reinstall PyTorch:

    - **Linux:** PyTorch version 2.3.1
    - **Windows:** PyTorch version 2.5.1

- Users can select the correct index URL for their CUDA version based on the `PyTorch getting started page <https://pytorch.org/get-started/locally>`_

**Example on a Linux x86 system:** ::

    $ pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121

*************************
Managing your environment
*************************

We recommend installing NiChart Project within a dedicated environment. Users can create an environment using Mamba (please see `Mamba Installation Guide <https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html>`_).

**Example on a Linux x86 system:** ::

    $ wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh
      bash Mambaforge-Linux-x86_64.sh
      mamba create -c conda-forge -c bioconda -n NCP_env python=3.12
      mamba activate NCP_env
      git clone https://github.com/CBICA/NiChart_Project.git
      pip install -r requirements.txt
