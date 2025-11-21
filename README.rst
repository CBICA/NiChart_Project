NiChart: Neuro-imaging Chart
============================

.. image:: https://img.shields.io/badge/Documentation-Read_the_Docs-blue
    :target: https://cbica.github.io/NiChart_Project

.. image:: https://img.shields.io/badge/Website-NeuroImagingChart-orange
    :target: https://neuroimagingchart.com
    
.. image:: https://img.shields.io/badge/GitHub-CBICA/NiChart_Project-green
    :target: https://github.com/CBICA/NiChart_Project

About
-----

*NiChart* is a novel AI-powered neuroimaging platform with tools for computing a dimensional chart from multi-modal MRI data. *NiChart* provides end-to-end pipelines from raw DICOM data to advanced
AI biomarkers, allowing to map a subject’s MRI images into personalized measurements, along with
reference distributions for comparison to a broader population.

.. image:: https://raw.githubusercontent.com/CBICA/NiChart_Project/refs/heads/ge-dev/resources/images/NiChart_Flowchart_v2.svg
  :alt: NiChart Flowchart

This repo contains the NiChart application front-end, which ties together all individual tools in the NiChart ecosystem and provides an easy-to-use interface for processing your data. For other tools, see the linked repositories.

The Basics
----------

The development of NiChart is guided by several core principles:

 1. Enabling **near real-time image processing and analysis** through advanced methods.

 2. Facilitating the **continuous integration** of **cutting-edge methods** for extracting novel **AI biomarkers** from neuroimaging data.

 3. Ensuring robust and reliable results through **extensive data training and validation** on large and diverse training datasets.

 4. Providing user-friendly tools for **visualization and reporting**.

 5. Developing a deployment strategy that enables **easy access** for users with varying technical expertise and hardware resources.

Running NiChart
---------------

We provide both a locally installable **desktop application** and a **cloud-based application**. 

The `NiChart cloud application <https://neuroimagingchart.com/portal>`_, hosted via Amazon Web Services (AWS), deploys scalable infrastructure which hosts the *NiChart* tools as a standard web application accessible via the user’s web browser. **No payment or installation is needed to use the tool**. You don't need any special hardware to run this.

However, as a web application, NiChart Cloud requires you to upload your data to the private cloud-based NiChart server for us to process it. **We do not access or use your data** for any other purpose than to run your requested processing and/or provide support to you as a user, and we regularly automatically delete user data after inactivity. However, we recognize that data privacy agreements and related concerns may nevertheless restrict use of the cloud application. If that applies to you, we suggest that you install the desktop application. We provide detailed installation instructions on the `Installation page <./INSTALLATION.md>`_. 

The NiChart front-end desktop application currently supports Windows and Linux. Windows has been tested on recent versions of Windows 10 and Windows 11. Linux has been tested on Ubuntu 24.04 but should work on other distributions. An NVIDIA GPU supporting CUDA is required to run the algorithms. If you need to run the algorithms with alternative GPU hardware or on CPU, please use the standalone tools.

**Want to switch between versions?** The cloud and desktop applications are unified at the code level through the use of the Python library `Streamlit <https://streamlit.io>`_. Consequently, the user experience is nearly identical between the cloud and desktop applications. 

Looking for specific NiChart tools?
-------------------

If you're looking for the individual NiChart structural tools, please see their individual repos:

NiChart_DLMUSE  [`GitHub <https://github.com/CBICA/NiChart_DLMUSE>`_] [`Docker Hub <https://hub.docker.com/repository/docker/cbica/nichart_dlmuse>`_] - Fast brain segmentation via deep learning

NiChart_DLWMLS [`GitHub <https://github.com/CBICA/NiChart_DLWMLS>`_] [`Docker Hub <https://hub.docker.com/repository/docker/cbica/dlwmls_wrapped>`_]  - Fast white matter lesion segmentation via deep learning

SPARE scores  [`GitHub <https://github.com/CBICA/spare_score>`_] [`Docker Hub <https://hub.docker.com/repository/docker/cbica/nichart_spare_score>`_] - ML-based prediction/scoring for variables of clinical interest

CCL-NMF prediction  [`GitHub <https://github.com/CBICA/CCL_NMF_Prediction>`_] [`Docker Hub <https://hub.docker.com/repository/docker/cbica/ccl_nmf_prediction_wrapped>`_]  - Lightweight estimation of CCL-NMF loading coefficients

ComBatFam Harmonization [`GitHub <https://github.com/PennSIVE/ComBatFam_Pipeline>`_] [`Docker Hub <https://hub.docker.com/repository/docker/cbica/combatfam_harmonize_dlmuse>`_]  - Dataset harmonization tools

SurrealGAN / PredCRD  [`GitHub <https://github.com/CBICA/PredCRD>`_] [`Docker Hub <https://hub.docker.com/repository/docker/cbica/surrealgan_predcrd_wrapped>`_]  - Predict continuous representation of disease along 5 principal dimensions


Quick Links
-----------

.. image:: https://img.shields.io/badge/Research-AIBIL-blue
    :target: https://aibil.med.upenn.edu/research
    :alt: AIBIL Research

.. image:: https://img.shields.io/badge/YouTube-%23FF0000.svg?style=for-the-badge&logo=YouTube&logoColor=white
    :target: https://www.youtube.com/@NiChart-UPenn
    :alt: YouTube

.. image:: https://img.shields.io/twitter/url/https/twitter.com/NiChart_AIBIL.svg?style=social&label=Follow%20%40NiChart_AIBIL
    :target: https://x.com/NiChart_AIBIL
    :alt: Twitter

© 2025 CBICA. All Rights Reserved.
