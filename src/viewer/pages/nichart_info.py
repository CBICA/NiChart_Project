
import utils.utils_pages as utilpg
import os
import streamlit as st
from utils.utils_styles import inject_global_css
from utils.utils_logger import setup_logger
import utils.utils_session as utilses

logger = setup_logger()
logger.debug("--- STARTING: Info ---")

inject_global_css()

utilpg.set_global_style()

st.set_page_config(page_title="NiChart", layout="wide")

imgdir = os.path.join(st.session_state.paths['resources'], 'images', 'nichart_logo')

def imgfile_to_data(filepath):
    import base64
    with open(filepath, "rb") as f:
        data = f.read()
        encoded = base64.b64encode(data)
    return "data:image/png;base64," + encoded.decode("utf-8")

tabA, tabB = st.tabs(
    ["Quick Start", "Learn More"],
    on_change='rerun',
)

imgdir =  os.path.join(st.session_state.paths['resources'], 'images', 'nichart_logo')
imgdir_base = os.path.join(st.session_state.paths['resources'], 'images')

if tabA.open:
    with tabA:

        st.markdown(
            '''
            ##### Getting Started with NiChart
            
            **NiChart** provides normative brain aging charts derived from large lifespan datasets. Here is how to get the most out of it:
            
            ###### :material/show_chart:  :blue[**Explore reference charts**]
            
            > No data required. Go to **View Charts → Reference Charts** to browse age-related centile curves for all brain regions and biomarkers.
            
            ###### :material/play_circle: :blue[**Process your data**]
            
            > Select **Process a Single Subject** or **Process a Dataset**. Both workflows cover the same pipelines; choose based on your data.
                       
            > Upload your images, select a pipeline, validate the expected inputs, then run. Results are saved to your project folder. Output files can be downloaded as a ZIP archive.
            
            ###### :material/person: :blue[**View personalized charts**]
            
            > Once processing is complete, go to **View Charts → My Charts** to overlay your data on the normative centile curves.
            
            ###### :material/summarize: :blue[**Generate reports**]
             
            > Use **View Charts → My Reports** to create subject-level PDF reports showing  scores for key brain regions.

            ###### :material/upload_file: :blue[**Skip processing**]
             
            > If you already have NiChart-format CSV results, upload them directly under **My Data**. Column names and primary keys must match the expected format (see the data validation panel for details)

            '''
        )
if tabB.open:
    with tabB:
        tabB1, tabB2, tabB3 = st.tabs(
            ["Methods", "Data", "Links"],
            on_change='rerun',
        )
        if tabB1.open:
            with tabB1:
                tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
                    ["NiChart", "MRI Segmentation", "AI Biomarkers", "Brain Aging Dimensions", "Abnormality Maps", "DKGP Biomarker Trajectories (WIP)", "Glioblastoma Heterogeneity Measures", "Glioblastoma Recurrence Maps"],
                    on_change='rerun',
                )

                if tab1.open:
                    with tab1:
                        st.markdown("<h5 style='color:#3a3a88;'>NiChart Overview</h5>", unsafe_allow_html=True)
                        with st.container(horizontal=True, border=False):
                            st.image(imgfile_to_data(os.path.join(imgdir, 'nichart_logo_v2_img1_v2.png')))
                            st.markdown(
                                """
                                **NeuroImaging Chart of AI-based Imaging Biomarkers**

                                A framework to:

                                - Process MRI images
                                - Harmonize scans to reference datasets
                                - Apply and contribute machine learning models
                                - Derive individualized neuroimaging biomarkers
                                """
                            )

                if tab2.open:
                    with tab2:
                        st.markdown("<h5 style='color:#3a3a88;'>MRI Segmentation</h5>", unsafe_allow_html=True)
                        with st.container(horizontal=True, border=False):
                            st.image(imgfile_to_data(os.path.join(imgdir, 'nichart_logo_v2_img4_v2.png')))
                            st.markdown(
                                """
                                **Segmentation of Brain Anatomy**

                                NiChart integrates DL-based models to calculate:

                                - **DLICV:** Intra-cranial volume estimation
                                - **DLMUSE:** Region of interest segmentation https://pubmed.ncbi.nlm.nih.gov/26679328
                                - **DLWMLS:** WM lesion segmentation https://pubmed.ncbi.nlm.nih.gov/26679328
                                """
                            )

                if tab3.open:
                    with tab3:
                        st.markdown("<h5 style='color:#3a3a88;'>AI Biomarkers</h5>", unsafe_allow_html=True)
                        with st.container(horizontal=True, border=False):
                            st.image(imgfile_to_data(os.path.join(imgdir, 'nichart_logo_v2_img3_v2.png')))

                            st.markdown(
                                """
                                **Imaging biomarkers of brain aging and disease**

                                NiChart uses raw T1 images and/or derived features to compute a set of predictive biomarkers (SPARE scores - Spatial Patterns of Abnormalities reflect structural variability in the brain associated with a given task)

                                - **SPARE-BA:** An individualized index reflecting the brain age

                                - **DeepSPARE-BA:** An individualized index reflecting the brain age and derived directly from raw T1 scan

                                - **SPARE-AD:** An individualized index quantifying the presence and severity of Alzheimer's disease (AD)-like patterns of atrophy in the brain (https://pubmed.ncbi.nlm.nih.gov/19416949/)

                                - **SPARE-CVMs:** The cardiometabolic risk models (smoking, obesity, hypertension, and diabetes) https://www.nature.com/articles/s41467-025-57867-7

                                - Other SPARE disease models reflect the specific conditions for depression (**SPARE-Depression**) and psychosis (**SPARE-Psychosis**)
                                """
                            )

                if tab4.open:
                    with tab4:
                        st.markdown("<h5 style='color:#3a3a88;'>Brain Aging Dimensions</h5>", unsafe_allow_html=True)
                        with st.container(horizontal=True, border=False):
                            st.image(imgfile_to_data(os.path.join(imgdir, 'nichart_logo_v2_img5_v2.png')))
                            st.markdown(
                                """
                                **Semi-supervised ML models of brain aging heterogeneity**

                                Brain aging dimensions reflect continuous latent representations of structural patterns associated with aging.

                                - **Surreal-GAN R-indices:** https://pubmed.ncbi.nlm.nih.gov/39147830/

                                - **CCLNMF indices:** Coupled Cross-Sectional and Longitudinal Non-Negative Matrix Factorization 

                                **Note:** Surreal-GAN and CCL-NMF indices in NiChart were obtained using a knowledge distillation method to train a tabular transformer with four encoder layers to predict the original indices
                                """
                            )

                if tab5.open:
                    with tab5:
                        st.markdown("<h5 style='color:#3a3a88;'>Abnormality Maps</h5>", unsafe_allow_html=True)
                        with st.container(horizontal=True, border=False):
                            st.image(imgfile_to_data(os.path.join(imgdir, 'nichart_logo_v2_img6_v2.png')))
                            st.markdown(
                                """
                                **Voxelwise CSF abnormality maps to quantify regional brain atrophy***

                                Voxelwise abnormality maps quantify how much each brain region deviates from a normative aging model, highlighting localized tissue loss or expansion

                                - Abnormality maps were derived using mass-preserving tissue density measures (**RAVENS maps**), enabling precise regional comparisons of gray matter, white matter, and CSF volumes.

                                - Combining RAVENS with CSF-based abnormality maps yields a spatial fingerprint of structural vulnerability, showing where tissue density differs from healthy controls at the voxel level.

                                - These maps allow subject-level interpretation, enabling visualization of individual neuroanatomical abnormalities, not just group averages.
                                """
                            )
                if tab6.open:
                    with tab6:
                        with st.container(horizontal=True, border=False):
                            st.markdown("<h5 style='color:#3a3a88;'>DKGP Biomarker Trajectories (WIP)</h5>", unsafe_allow_html=True)
                            st.markdown(
                                """
                                Personalized biomarker trajectory forecasting uses Deep Kernel Gaussian Processes to predict how individual biomarkers will evolve over time.

                                - Trajectories are modeled using a deep kernel learning framework that combines neural feature extraction with Gaussian Process regression, capturing complex nonlinear patterns from cross-sectional neuroimaging data, demogrpahic and clinical variables.

                                - The population-level model (p-DKGP) is trained across large multi-cohort datasets, and delivers individualized forecasts from cross-sectional data alone.
                                """
                            )
                if tab7.open:
                    with tab7:
                        with st.container(horizontal=True, border=False):
                            col1, col2 = st.columns(0.7, 0.3)
                            with col2:
                                st.image(imgfile_to_data(os.path.join(imgdir_base, 'ithresolvegbm_figure.png')))
                            with col1:
                                st.markdown('''
                                            **Decode GBM heterogeneity from bulk DNA methylation data.**

                                            ITHresolveGBM is a deconvolution framework that infers the cellular composition of glioblastoma samples from standard DNA methylation profiles. It estimates tumor purity, quantifies immune, glial, and neuronal microenvironmental components, as well as the malignant component and the ratio of stem-like and differentiated states within it. It is most helpful when you have bulk DNA methylation data but would like to infer the cellular composition within each sample.
                                            
                                            **Please cite:**  
                                            Silverbush et al., *A cellular epigenetic classification system for glioblastoma, Neuro-Oncology*, 2026.

                                            **Successful applications include:**
                                            - In ongoing work, ITHresolveGBM outputs are used to compute cell state-informed methylation classes (MCcsi), enabling patient stratification by predominant cellular states in primary and recurrent glioblastoma. These classes inform molecular features, prognosis, and treatment-associated vulnerabilities in the context of standard of care, iVEGFA, iTOP2A, and iCDK4, establishing transcriptional cell state heterogeneity as a readily implementable tumor biomarker for glioblastoma treatment (Ito et al.).
                                            - Applied to high-grade gliomas, ITHresolveGBM revealing a high-neural glioblastoma state enriched for malignant stem/progenitor-like programs and reduced immune infiltration. This high-neural state was associated with increased neuron-to-glioma interactions, greater tumor invasiveness, and worse patient outcomes (Drexler et al., *Nature Medicine*, 2024).
                                            - Applied to multi-region GBM methylation profiles, it was used to show that the proportion of stem-like malignant cells is preserved across spatially distinct regions of the same tumor. A higher stem-like fraction was associated with worse survival, reduced benefit from MGMT promoter methylation, and radiomethylomic signatures detectable from preoperative MRI (Matsumoto et al., *Neuro-Oncology*, 2025).
                                            
                                            Wrapper built by Jakob Ito.
                                            ''')
                            
                if tab8.open:
                    with tab8:
                        with st.container(horizontal=True, border=False):
                            st.markdown("Glioblastoma recurrence probability maps are generated by an ensemble of ML models trained and validated using multi-model MRI scans across the international ReSPOND consortium. The various  models capture imaging patterns associated with high tumor infiltration and increased likelihood of recurrence at different spatial scales to generate a 3D probability map of recurrence risk within the peritumoral edema. Large, connected regions predicted to be high risk are identified as potential treatment targets.")
                            st.image(imgfile_to_data(os.path.join(imgdir_base, 'gbm_recurrence_maps_figure.png')))
        if tabB2.open:
            with tabB2:
                st.markdown(
                    '''
                    - NiChart Reference Dataset is a large and diverse collection from multiple MRI studies, created as part of the ISTAGING project to develop a system for identifying imaging biomarkers of aging and neurodegenerative diseases.

                    - The dataset includes multi-modal MRI data, as well as carefully curated demographic, clinical, and cognitive variables from participants with a variety of health conditions.

                    - The reference dataset is used for training machine learning models and for creating reference distributions of imaging measures and signatures

                    - Users can compare their values to normative or disease-related reference distributions.            '''
                )
                st.image(
                    os.path.join(
                        st.session_state.paths['resources'], 'images', 'nichart_data.png'
                    ),
                    width=1200
                )

        if tabB3.open:
            with tabB3:
                st.info('Links to Project Page and GitHub will be here soon ...')


# Home button
utilpg.navig_home()

# Show session state vars
if st.session_state.mode == 'debug':
    utilses.disp_session_state()
