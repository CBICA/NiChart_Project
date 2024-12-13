##########
Components
##########

NiChart is designed to integrate independent image processing and analysis pipelines. The following sections detail these pipelines.

*****************
Current Pipelines
*****************

These pipelines are currently active and accessible within the NiChart Project

====================================
1. sMRI Biomarkers of Disease and Aging
====================================

Neuroimaging pipeline for computing AI biomarkers of disease and aging from T1-weighted MRI scans. The pipeline applies the following steps for processign and analysis.

------------
Segmentation
------------

`DLMUSE <https://neuroimagingchart.com/components/#Image%20Processing>`_: Rapid and accurate **brain anatomy segmentation**

.. image:: https://github.com/CBICA/NiChart_Project/blob/031d1cafc5091eb514511ee25af189d5f0b5ac56/resources/images/dlicv%2Bdlmuse_segmask.png
   :alt: DLMUSE

-------------
Harmonization
-------------

`COMBAT <https://neuroimagingchart.com/components/#Harmonization>`_: **Statistical data harmonization** of ROI volumes to `reference data <https://neuroimagingchart.com/components/#Reference%20Dataset>`_

.. image:: https://raw.githubusercontent.com/CBICA/NiChart_Project/refs/heads/ge-dev/resources/images/combat_agetrend.png
   :alt: COMBAT

------------------------
Supervised ML Biomarkers
------------------------

`SPARE-AD and SPARE-Age indices <https://neuroimagingchart.com/components/##Machine%20Learning%20Models>`_: AI biomarkers of **Alzheimer's Disease and Aging** related brain atrophy patterns

.. image:: https://raw.githubusercontent.com/CBICA/NiChart_Project/refs/heads/ge-dev/resources/images/sparead%2Bage.png
  :alt: SPARE-AD and SPARE-Age indices

`SPARE-CVR indices <https://alz-journals.onlinelibrary.wiley.com/doi/abs/10.1002/alz.067709>`_: AI biomarkers of brain atrophy patterns associated with **Cardio-Vascular Risk Factors**

.. image:: https://raw.githubusercontent.com/CBICA/NiChart_Project/refs/heads/ge-dev/resources/images/sparecvr.png
  :alt: SPARE-CVR indices, Govindarajan, S.T., et. al., Nature Communications, 2024

-----------------------------
Semi-supervised ML Biomarkers
-----------------------------

 - `SurrealGAN indices <https://www.nature.com/articles/d41586-024-02692-z>`_: Data-driven phenotyping of brain aging, **5 Brain Aging Subtypes**

.. image:: https://raw.githubusercontent.com/CBICA/NiChart_Project/refs/heads/ge-dev/resources/images/sgan1.jpg
   :alt: SurrrealGAN indices

====================================
2. WM Lesion Segmentation
====================================

Neuroimaging pipeline for segmenting white matter lesions on FLAIR MRI scans.

`DLWMLS <https://neuroimagingchart.com/components/#Image%20Processing>`_: Rapid and accurate **white matter lesion segmentation**

.. image:: https://github.com/CBICA/NiChart_Project/blob/031d1cafc5091eb514511ee25af189d5f0b5ac56/resources/images/dlwmls.png
   :target https://github.com/CBICA/NiChart_DLWMLS
   :alt: DLWMLS

*****************
Under Development
*****************

These pipelines are planned for integration in future NiChart releases.

====================================
1. DTI Biomarkers of Disease and Aging
====================================

====================================
2. fMRI Biomarkers of Disease and Aging
====================================
