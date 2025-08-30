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

The Basics
----------

The development of nichart is guided by several core principles:

 1. Enabling **near real-time image processing and analysis** through advanced methods.

 2. Facilitating the **continuous integration** of **cutting-edge methods** for extracting novel **AI biomarkers** from neuroimaging data.

 3. Ensuring robust and reliable results through **extensive data training and validation** on large and diverse training datasets.

 4. Providing user-friendly tools for **visualization and reporting**.

 5. Developing a deployment strategy that enables **easy access** for users with varying technical expertise and hardware resources.

Running NiChart
---------------


We provide both a locally installable **desktop application** and a **cloud-based application**. 

`NiChart cloud application <https://neuroimagingchart.com/portal>`_, hosted via Amazon Web Services (AWS), deploys scalable infrastructure which hosts the *NiChart* tools as a standard web application accessible via the user’s web browser. **No install needed**, but it requires you to upload your data to the cloud-based NiChart server for us to process it. We do not access or use your data for any other purpose than to run your requested processing and/or provide support to you, and we regularly delete user data after inactivity. However, we recognize that data privacy agreements and related concerns may nevertheless restrict use of the cloud application. If that applies to you, we suggest that you install the desktop application. Below we provide detailed installation instructions.

The cloud and desktop applications are unified at the code level through the use of the Python library `Streamlit <https://streamlit.io>`_. Consequently, the user experience is nearly identical between the cloud and desktop applications.

**Desktop installation**: You have two options for installing NiChart locally as an application on your computer. Both options currently require `Docker <https://www.docker.com/get-started/>`_ to be installed, as this greatly simplifies deployment and distribution of our algorithms without requiring extensive dependency management. Follow the instructions to install Docker (or Docker Desktop, on Windows/Mac) for your platform, then restart your device before continuing. We recommend having at least 20 GB of free space on your device before installing NiChart.

**Installation Option 1**: Use Docker to run NiChart itself. This avoids the need for any Python installation, but may take up a little more space on your drive and requires a small amount of configuration. Please follow all steps carefully.

First, if you're on Windows, open Docker Desktop. You can do this from the start/search menu or by clicking the Desktop shortcut if you selected that during installation. You should go into the settings using the gear icon on the top right, go to "General", and enable the settings "Use the WSL 2 based engine" and "Expose daemon on tcp://localhost:2375 without TLS" if they aren't already enabled (they might require you to restart). You should also see a green indicator on the bottom left which says "Engine running". If it's yellow, you need to wait for the service to start. Otherwise, you may need to troubleshoot your installation. 

Make sure you are connected to the internet in order to download the application. Then, open a terminal.

(On Windows, search "terminal", open the application that looks like a black box with a white ">_" in it. At the top of the window that appears will be a tab indicating Windows Powershell. Click the down arrow next to that tab to expand your terminal options, and select Ubuntu. A new terminal will open in a different color and you should see something like "root@username:~#". Stay on this tab for the rest of the instructions.)

Run the following command:

``docker pull cbica/nichart:07032025``

** TO BE FINISHED **





**Option 2**: Install Python locally and run the NiChart application. The NiChart application will attempt to communicate automatically with Docker on your machine.

** TO BE FINISHED **

**Can't use Docker?** We aim to soon provide compatibility with Singularity/Apptainer runtimes for users in computing environments where Docker is disallowed. Please check in regularly for updates.

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
