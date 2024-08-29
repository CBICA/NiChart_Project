
Installation
============

``pNet`` is installable via ``pip``:

.. code-block:: bash

	pip install fmripnet

``pNet`` is accessbile via ``docker``:

.. code-block:: bash
        
        docker pull mldataanalytics/fmripnet:latest

or:

.. code-block:: bash

        docker pull ghcr.io/mldataanalytics/fmripnet:latest

run:

.. code-block:: bash

        docker run mldataanalytics/fmripnet -h

Alternatively, you can install the most up-to-date version of from GitHub:

.. code-block:: bash

	git clone https://github.com/MLDataAnalytics/pNet.git
	cd pNet
        conda env create --name fmripnet -f environment_pnet.yml
	pip install . 

Note that ``pnet`` requires Python 3.8+ and some key dependencies:
  - h5py
  - mesalib
  - nibabel
  - numpy
  - pandas
  - pip
  - python==3.8.13
  - pytorch==2.1.0
  - scikit-image
  - scikit-learn
  - scipy
  - vtk>=9.2=*osmesa*
  - ggplot
  - matplotlib
  - plotnine
  - statsmodels
  - surfplot
  - tomli

Support:

If you encounter problems or bugs with pNet, or have questions or improvement suggestions, please feel free to get in touch via the Github issues: https://github.com/MLDataAnalytics/pNet/issues.

Previous versions:

 - Matlab and Python: https://github.com/MLDataAnalytics/pNet_Matlab
 - Matlab: https://github.com/MLDataAnalytics/Collaborative_Brain_Decomposition
 - GIG-ICA: https://www.nitrc.org/projects/gig-ica/

Other useful packages:

 - brainspace: https://brainspace.readthedocs.io/en/latest/index.html
 - matplotlib: https://matplotlib.org/
 - numpy: https://numpy.org/
 - nibabel: https://nipy.org/nibabel/
 - vtk: https://vtk.org/
 - nilearn: https://nilearn.github.io/index.html
 - neuromaps: https://netneurolab.github.io/neuromaps/
