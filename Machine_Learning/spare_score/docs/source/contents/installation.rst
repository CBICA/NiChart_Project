************
Installation
************

spare_scores can be installed using `pip`. We suggest that users install the package in a Conda environment or Python3 virtual environment:

**1. Installation in a Conda environment using pip**

   .. code-block:: console

      conda create -n spare python=3.8
      conda activate spare
      conda install pip
        
      pip install spare_scores

**2. Installation in a Python3 virtual environment using pip**

   .. code-block:: console
        
      python3 -m venv env spare
      source spare/bin/activate
        
      pip install spare_scores

**3. Installation in a Conda environment from Github repository**

   .. code-block:: console

      git clone https://github.com/CBICA/spare_score.git
      cd spare_score
      pip install .