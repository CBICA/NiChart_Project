pNet
====

pNet is a Python package for computing personalized, sparse,
non-negative large-scale functional networks from functional magnetic
resonance imaging (fMRI) data, particularly resting state fMRI data. The
personalized functional networks are comparable across subjects while
maintaining subject specific variation, reflected by their improved
functional coherence compared with their group-level counterparts. The
computation of personalized functional networks is accompanied by
quality control with visualization and quantification of their spatial
correspondence and functional coherence in reference to their
group-level counterparts.

.. figure::
   https://github.com/user-attachments/assets/b45d02a1-2c82-43b5-b7d5-42fc38a7b298
   :alt: image


Getting started
---------------

Follow the Installation Instructions to install pNet, and then check out
the Tutorials and
`Examples <https://github.com/MLDataAnalytics/pNet/tree/main/src/pnet/examples>`__
to learn how to get up and running! 

1. Download pNet
~~~~~~~~~~~~~~~~

::

   git clone https://github.com/MLDataAnalytics/pNet.git

2. Create a new conda environment for pNet
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   cd pNet
   conda env create --name fmripnet -f environment_pnet.yml

3. Install pNet
~~~~~~~~~~~~~~~

::

   conda activate fmripnet
   pip install .
   # or pip install fmripnet

Script usages
~~~~~~~~~~~~~

1. Prepare data
^^^^^^^^^^^^^^^

::

   1) a number of preprocessed fMRI scans that have been spatially aligned to a template space,
   2) a mask image for excluding voxels/vertices of uninterest,
   3) a brain template image/surface for visualization
   4) a script can be found in cli folder for preparing the brain template data

2. Run the computation (examples can be found in examples folder)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

   5) a script (fmripnet.py) can be found in cli folder  for running the computation, supplied with a configuration file (*.toml) for setting the input and output information
      run "python fmripnet.py -h " to get help information
      run "python fmripnet.py -c a_config.toml" to start the computation
      run "python fmripnet.py -c a_config.toml --hpc qsub" to start the computation on a HPC cluster with qsub

Code examples and usages
~~~~~~~~~~~~~~~~~~~~~~~~

.. _prepare-data-1:

1. Prepare data
^^^^^^^^^^^^^^^

::

   1) a number of preprocessed fMRI scans that have been spatially aligned to a template space,
   2) a mask image for excluding voxels/vertices of uninterest,
   3) a brain template image/surface for visualization

2. Setup the computation
^^^^^^^^^^^^^^^^^^^^^^^^

::

   1) the number of functional networks,
   2) the output folder information,
   3) optional parameters

3. Example code:
^^^^^^^^^^^^^^^^

::

   import pnet

   # create a txt file of fMRI scans, each line with a fMRI scan 
   file_scan = 'sbj_lst.txt'
   # create a brain template file consisting of information of the mask image and the brain template for visualization or use a template that is distributed with the package) 
   file_Brain_Template = pnet.Brain_Template.file_MNI_vol

   # Setup
   # data type is volume
   dataType = 'Volume'
   # data format is NIFTI, which stores a 4D matrix
   dataFormat = 'Volume (*.nii, *.nii.gz, *.mat)'
   # output folder
   dir_pnet_result = 'Test_FN17_Results'

   # number of FNs
   K = 17

   # Setup number of scans loaded for each bootstrap run for estimating group functional networks
   sampleSize = 100 # The number should be no larger than the number of available fMRI scans. A larger number of samples can improve the computational robustness but also increase the computational cost.  Recommended: >=100
   # Setup number of runs for bootstraps
   nBS = 50         # A larger number of run can improve the computational robustness but also increase the computational cost. recommended: >=10
   # Setup number of time points for computing group FNs with bootstraps
   nTPoints = 200   # The number should be no larger than the number of available time points of the fMRI scans. A larger number of samples can improve the computational robustness but also increase the computational cost.  If not set, all available time points will be used if smaller than 9999.

   # Run pnet workflow
   pnet.workflow_simple(
           dir_pnet_result=dir_pnet_result,
           dataType=dataType,
           dataFormat=dataFormat,
           file_scan=file_scan,
           file_Brain_Template=file_Brain_Template,
           K=K,
           sampleSize=sampleSize,
           nBS=nBS,
           nTPoints=nTPoints
       )

Support
-------

If you encounter problems or bugs with pNet, or have questions or
improvement suggestions, please feel free to get in touch via the
`Github issues <https://github.com/MLDataAnalytics/pNet/issues>`__.

Previous versions:
------------------

 - Matlab and Python: https://github.com/MLDataAnalytics/pNet_Matlab

 - Matlab: https://github.com/MLDataAnalytics/Collaborative_Brain_Decomposition

 - GIG-ICA: https://www.nitrc.org/projects/gig-ica/
