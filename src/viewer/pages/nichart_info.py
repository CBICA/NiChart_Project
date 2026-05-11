
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

def show_short_desc(title):
    if title == "NiChart":
        st.markdown("Neuroimaging Chart of **AI-based** imaging biomarkers")

        pass
    if title == "MRI Segmentation":
        st.markdown("Fast **deep-learning** segmentation of **healthy** and **pathological** anatomy")
        pass
    if title == "AI Biomarkers":
        st.markdown("AI-based **imaging biomarkers** quantifying **brain aging** and **neurodegeneration**")
        pass
    if title == "Brain Aging Dimensions":
        st.markdown("**Data-driven** brain aging indices capturing **distinct patterns** of **aging-related atrophy**")
        pass
    if title == "Abnormality Maps":
        st.markdown("Voxelwise CSF abnormality maps quantifying regional brain atrophy.")
        pass
    if title == "DKGP Biomarker Trajectories (Work in Progress)":
        st.markdown("Predict evolution of biomarkers over time.")

def show_full_desc(title):
    if title == "NiChart":
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
    if title == "MRI Segmentation":
        st.markdown(
            """
            **Segmentation of Brain Anatomy**
            
            NiChart integrates DL-based models to calculate:
            
            - **DLICV:** Intra-cranial volume estimation 
            - **DLMUSE:** Region of interest segmentation https://pubmed.ncbi.nlm.nih.gov/26679328
            - **DLWMLS:** WM lesion segmentation https://pubmed.ncbi.nlm.nih.gov/26679328
            """
        )
            
    if title == "AI Biomarkers":
        st.markdown(
            """
            **Supervised ML models of brain aging and disease**
            
            NiChart uses raw T1 images and/or derived features to compute a set of predictive biomarkers (SPARE scores - Spatial Patterns of Abnormalities reflect structural variability in the brain associated with a given task)
            
            - **SPARE-BA:** An individualized index reflecting the brain age
            
            - **DeepSPARE-BA:** An individualized index reflecting the brain age and derived directly from raw T1 scan

            - **SPARE-AD:** An individualized index quantifying the presence and severity of Alzheimer’s disease (AD)-like patterns of atrophy in the brain (https://pubmed.ncbi.nlm.nih.gov/19416949/
            
            - **SPARE-CVMs:** The cardiometabolic risk models (smoking, obesity, hypertension, and diabetes) https://www.nature.com/articles/s41467-025-57867-7

            - Other SPARE disease models reflect the specific conditions for depression (**SPARE-Depression**) and psychosis (**SPARE-Psychosis**)
            """
        )

    if title == "Brain Aging Dimensions":
        st.markdown(
            """
            **Semi-supervised ML models of brain aging heterogeneity**
            
            Brain aging dimensions reflect continuous latent representations of structural patterns associated with aging. 
            
            - **Surreal-GAN R-indices:** The R-index reflects the severity of individualized brain changes along multiple dimensions, potentially reflecting the stage of a mixture of underlying neuropathological and biological processes https://pubmed.ncbi.nlm.nih.gov/39147830/
            
                ***R1:*** subcortical atrophy, mainly concentrated in the caudate and putamen
            
                ***R2:*** focal medial temporal lobe (MTL) atrophy
            
                ***R3:*** parieto-temporal atrophy, including that in middle temporal gyrus, angular gyrus and middle occipital gyrus
            
                ***R4:*** diffuse cortical atrophy in medial and lateral frontal regions, as well as superior parietal and occipital regions
            
                ***R5:*** perisylvian atrophy centered around the insular cortex
            
            - **CCLNMF indices:** Coupled Cross-Sectional and Longitudinal Non-Negative Matrix Factorization identifies dominant patterns of brain aging by jointly modeling baseline (cross-sectional) and follow-up (longitudinal) MRI data.
            
                Cross-sectional maps capture cumulative aging differences across individuals, while longitudinal maps quantify subject-specific change rates. These maps are jointly decomposed via NMF into shared spatial components and individual loadings, providing continuous measures of distinct aging patterns.
            
            **Note:** Surreal-GAN and CCL-NMF indices in NiChart were obtained using a knowledge distillation method to train a tabular transformer with four encoder layers to predict the original indices


            """
        )

        st.markdown('''

''')
    if title == "Abnormality Maps":
        st.markdown(
            """
            **CSF Abnormalities** ***(work in progress)***
            
            Voxelwise abnormality maps quantify how much each brain region deviates from a normative aging model, highlighting localized tissue loss or expansion
            
            - Abnormality maps were derived using mass-preserving tissue density measures (**RAVENS maps**), enabling precise regional comparisons of gray matter, white matter, and CSF volumes.
            
            - Combining RAVENS with CSF-based abnormality maps yields a spatial fingerprint of structural vulnerability, showing where tissue density differs from healthy controls at the voxel level.
            
            - These maps allow subject-level interpretation, enabling visualization of individual neuroanatomical abnormalities, not just group averages.
            """
        )
    if title == "DKGP Biomarker Trajectories (Work in Progress)":
        st.markdown(
            """
            Personalized biomarker trajectory forecasting uses Deep Kernel Gaussian Processes to predict how individual biomarkers will evolve over time.

            - Trajectories are modeled using a deep kernel learning framework that combines neural feature extraction with Gaussian Process regression, capturing complex nonlinear patterns from cross-sectional neuroimaging data, demogrpahic and clinical variables.

            - The population-level model (p-DKGP) is trained across large multi-cohort datasets, and delivers individualized forcasts from cross-sectional data alone.
            """
        )

def card(title, image_path):
    with st.container(border=True, horizontal_alignment = 'left'):
        st.image(imgfile_to_data(image_path))
        with st.container(border=False, height = 200, horizontal_alignment = 'center'):
            st.markdown(f"#### {title}")
            show_short_desc(title)
        with st.popover("See More"):
            show_full_desc(title)

with st.container(horizontal_alignment="center"):
    st.markdown("## What can I do with NiChart?")
with st.container(horizontal=True, horizontal_alignment="center"):
    cols = st.columns(5)

    with cols[0]: # NiChart
        card(title="NiChart",
             image_path=os.path.join(imgdir, 'nichart_logo_v2_img1_v2.png')
            )

    with cols[1]: # Segmentation
        card(title="MRI Segmentation",
             image_path=os.path.join(imgdir, 'nichart_logo_v2_img4_v2.png')
             )

    with cols[2]: # AI Biomarkers
        card(title="AI Biomarkers",
             image_path=os.path.join(imgdir, 'nichart_logo_v2_img3_v2.png')
             )

    with cols[3]: # Brain Aging Subtypes
        card(title="Brain Aging Dimensions",
             image_path=os.path.join(imgdir, 'nichart_logo_v2_img5_v2.png')
             )

    with cols[4]: # Voxelwise Abnormality Maps
        card(title="Abnormality Maps",
             image_path=os.path.join(imgdir, 'nichart_logo_v2_img6_v2.png')
             )

with st.container(horizontal=True, horizontal_alignment="center"):
    secondary_cols = st.columns(1)
    with secondary_cols[0]:
        card(title="DKGP Biomarker Trajectories (Work in Progress)",
             image_path=os.path.join(imgdir, 'nichart_logo_v2_img1_v2.png'))



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
                tab1, tab2, tab3, tab4, tab5 = st.tabs(
                    ["NiChart", "MRI Segmentation", "AI Biomarkers", "Brain Aging Dimensions", "Abnormality Maps"],
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
