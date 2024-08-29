import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import Head from 'next/head';
import Header from '../components/Layout/Header';
import Footer from '../components/Layout/Footer';
import Favicons from '../components/Favicons/Favicons';
import Sidebar from '../components/Components/Sidebar';
import styles from '../styles/Components.module.css'

const Components = () => {
  const [expandedSection, setExpandedSection] = useState('Reference Dataset');
  const router = useRouter();

  useEffect(() => {
    const handleHashChange = () => {
      const hash = decodeURIComponent(window.location.hash.replace('#', ''));
      setExpandedSection(hash || 'Reference Dataset');
    };
  
    handleHashChange(); // Call on initial render
    window.addEventListener('hashchange', handleHashChange);
  
    return () => {
      window.removeEventListener('hashchange', handleHashChange);
    };
  }, [router.events]);
  

  const contentBySection = {
    'Reference Dataset': (
      <>
        <div className={styles.section} id='ref_data_overview'>
          <div className={styles.headerFlex}>
            <img className={styles.brainImage} src="/images/Components/brain_curate.png"/>
            <h1 style={{ color: '#a11f25' }}>NiChart Reference Dataset</h1>
          </div>
          <p>
            NiChart Reference Dataset is a large and diverse collection of MRI images from multiple studies. It was created as part of the <a href="https://www.med.upenn.edu/cbica/imaging-consortia-increasing-sample-size-and-understanding-heterogeneity-in-health-and-disease.html">ISTAGING project</a> to develop a system for identifying imaging biomarkers of aging and neurodegenerative diseases. The dataset includes multi-modal MRI data, as well as carefully curated demographic, clinical, and cognitive variables from participants with a variety of health conditions.
            The reference dataset is a key component of NiChart for training machine learning models and for creating reference distributions of imaging measures and signatures, which can be used to compare NiChart values that are computed from the user data to normative or disease-related reference values.
          </p>
          <p>The following table is as of this version incomplete; it will be updated as soon as possible.</p>
          <img src="/images/Components/Reference_Data_Curation/Picture7.png" alt=""/>
          <p>
          <i>Table 1. Overview of studies that are part of the NiChart Reference Dataset</i>
          </p>
          
          <br></br><br></br><hr/><br></br><br></br>

          <div className={styles.subsection} id="RefVars">
            <h2>Demographics and Clinical Variables</h2>
            <p>
              The reference dataset includes a large number of samples from people of different ethnic groups, with a focus on older adults. This diversity is important because it allows to train machine learning models that are more accurate for people of diverse backgrounds.
            </p>
            <img src="/images/Components/Reference_Data_Curation/figISTAGData_1.png" alt=""/>
            <p>
            <i>Figure 1. Reference dataset demographics </i>
            </p>
            <p>
              The reference dataset contains data from individuals with various neuro-degenerative diseases. Disease subgroups were used to train machine learning models specifically tailored to each disease and to calculate disease-specific reference distributions.
            </p>
            <img src="/images/Components/Reference_Data_Curation/figISTAGData_2.png" alt=""/>
            <i>Figure 2. Examples of disease subgroups in the reference dataset</i>
          </div>
        </div>

      </>
    ),
    'Image Processing': (
      <>
        <div className={styles.section} id="img_proc_overview">
          <div className={styles.headerFlex}>
            <img className={styles.brainImage} src="/images/Components/brain_process.png"/>
            <h1 style={{ color: '#2152ad' }}>Image Processing</h1>
          </div>
          <p>NiChart Image Processing Pipelines is a set of tools that can be used to extract features from multi-modal MRI data. The pipelines are made up of individual components that can be installed and run independently. Users can choose the specific components that they need for their analyses. NiChart pipelines are available both as <a href="https://github.com/CBICA/NiChart_Project">installation packages</a> and <a href="https://hub.docker.com/repository/docker/aidinisg/nichart_dlmuse/general">software containers</a>. This ensures reproducibility across different platforms, and allows users to easily install and run the pipelines without having to worry about installing any additional software dependencies.</p>
        
          <br></br><br></br><hr/><br></br><br></br>

          <div className={styles.subsection} id="sMRIProcessing">
            <h2>Structural MRI</h2>

            <p style={{padding: "2%"}}>NiChart uses a combination of both established and state of the art techniques to extract imaging features that quantify both normal and abnormal brain structures. Alongside conventional atlas-based segmentation methods for segmenting intra-cranial volume (ICV), anatomical regions of interest (ROIs), and white matter lesions (WMLs), we also offer an alternative  parcellation method using non-negative matrix factorization for generating multi-resolution data-driven structural covariance components. Our atlas-based segmentation methods are facilitated by deep learning networks that enable rapid segmentation.</p>
            
            <h3>DLICV:</h3>
            <p>Deep Learning Intra Cranial Volume (<a href="https://github.com/CBICA/DLICV">DLICV</a>) is a new deep learning (DL)-based tool to accurately segment the intracranial volume (ICV) from a raw T1-weighted MRI image. It's easy to use, requires minimal data preprocessing, and is robust against various factors that can affect segmentation accuracy. DLICV specifically segments the overall cerebrospinal fluid (CSF) surrounding the brain, rather than just the brain tissues themselves, providing an ICV estimation that is not influenced by overall cortical atrophy due to aging or disease.</p>
            <img src="/images/Components/ImageProcessing/sMRI/dlicv_ex1.png" alt=""/>
            <i>Figure 1. Example segmentation using DLICV (green) for cases with significant cortical atrophy</i>

            <h3>DLMUSE:</h3>
            <p>Deep Learning MUlti-atlas region Segmentation utilizing Ensembles of registration algorithms and parameters (<a href="https://github.com/CBICA/DLMUSE">DLMUSE</a>) is a tool for automatically segmenting T1-weighted brain MRI scans. It is accurate, robust, easy to use, and fast. DLMUSE is built on a 3D convolutional neural network (CNN) architecture that has been extensively validated for various neuroimaging segmentation tasks. DLMUSE model was trained on a large and diverse training set, with ROI labels derived using a computationally intensive multi-atlas segmentation method.</p>

            <h3>DLWMLS:</h3>
            <p>Deep Learning White Matter Lesion Segmentation (DLWMLS) is a multi-modal segmentation method for segmenting white mater hyper-intensities (brain lesions) from T1-weighted and FLAIR MRI images. DLWMLS model was trained on a large and diverse training set, with semi-automatically segmented labels for lesions.</p>
            <img src="/images/Components/ImageProcessing/sMRI/dl_ex2.png" alt=""/>
            <i>Figure 2. Example segmentatation using DLICV, DLMUSE and DLWMLS</i>
            

            <h3>sopNMF</h3>
            <p>Stochastic Orthogonally Projective Non-negative Matrix Factorization (<a href="https://www.medrxiv.org/content/10.1101/2022.07.20.22277727v2" target="_blank">sopNMF</a>) is an algorithm for large-scale multivariate structural analysis of human brain data. Using sopNMF, the MuSIC atlas parcellates the human brain by structural covariance in MRI data over the lifespan and a wide range of disease populations, allowing to explore the phenotypic landscape and genetic architecture of the human brain. You can find out more in the <a href="https://github.com/anbai106/SOPNMF">Github repository</a> of the python implementation.</p>
            <img src="/images/Components/ImageProcessing/sMRI/bridgeport.png" alt=""/>
            <i>Figure 3. Multi-resolution MuSIC atlas parcellation</i>
            
          </div>

          <br></br><br></br><hr/><br></br><br></br>
          
          <div className={styles.subsection} id="DTIProcessing">
            <h2>Diffusion Tensor Imaging</h2>
            <h3>QSIPrep:</h3>
            <p><a href="https://qsiprep.readthedocs.io/en/latest/">QSIPrep</a> is a specialized software platform designed for the preprocessing of diffusion MRI datasets, ensuring the deployment of adequate workflows for the task. It primarily focuses on diffusion-weighted magnetic resonance imaging (dMRI), a pivotal method for non-invasively examining the organization of white matter in the human brain. QSIPrep stands out for its integrative nature, being compatible with nearly all dMRI sampling schemes, thus providing a broad spectrum of utility in diffusion image processing.</p>
            <p>The platform employs an automated approach, configuring pipelines for processing dMRI data. It adheres to a BIDS-app methodology for preprocessing, which encompasses a variety of modern diffusion MRI data types. The preprocessing pipelines generated by QSIPrep are automatic, accurately grouping, distortion correcting, motion correcting, denoising, and coregistering the data, among other operations, to ensure the integrity and quality of the processed images.</p>
            <img src="/images/Components/ImageProcessing/fMRI/qsiprep.png" alt=""/>
            <i>Figure 4. QSIPrep flowchart</i>
          </div>
            
          <br></br><br></br><hr/><br></br><br></br>

          <div className={styles.subsection} id="fMRIProcessing">
            <h2>Functional MRI</h2>
            <p>Functional MRI processing combines well-established and extensively validated tools for image preprocessing, feature extraction, and calculation of functional networks.</p>
            
            <h3>fMRIPrep:</h3>
            <p><a href="https://fmriprep.org/en/stable/">fMRIPrep</a> is a robust preprocessing pipeline tailored for the analysis of functional Magnetic Resonance Imaging (fMRI) data. The pipeline leverages a combination of well-regarded software packages including FSL, ANTs, FreeSurfer, and AFNI to ensure optimal software implementation for each preprocessing stage. Designed to minimize manual intervention, fMRIPrep facilitates a transparent workflow that enhances the reproducibility of fMRI data analyses. It is suited for handling both task-based and resting-state fMRI data, adapting to the nuances of different datasets to provide high-quality preprocessing without requiring manual intervention.</p>
            <p>fMRIPrep is a <a href="https://www.nipreps.org">NiPreps</a> (NeuroImaging PREProcessing toolS) application for the preprocessing of task-based and resting-state functional MRI (fMRI).</p>
            <img src="/images/Components/ImageProcessing/fMRI/fmriprep-21.0.0.png" alt=""/>
            <i>Figure 5. fMRIPrep flowchart</i>
          
            <h3>XCPEngine</h3>
            <p>The XCPEngine, or XCP imaging pipeline, is an open-source software package engineered for processing multimodal neuroimages. Utilizing a modular design, it integrates analytic routines from leading MRI analysis platforms like FSL, AFNI, and ANTs. This engine offers a configurable, modular, and agnostic platform for neuroimage processing and quality assessment, encapsulating a variety of high-performance denoising approaches while computing regional and voxelwise values for each modality. </p>
            <img src="/images/Components/ImageProcessing/fMRI/qsiprep2.png" alt=""/>
            <i>Figure 6. XCPEngine flowchart</i>

            <h3>pNet</h3>
            <p>Personalized Functional Network Modeling (<a href="https://github.com/YuncongMa/pNet/tree/main">pNet</a>) is designed to provide a user-friendly interface to perform personalized functional network (pFN) computation and visualization. </p>
            <p>It is open-source, cross-platform, and expandable. The toolbox is built with support for MATLAB and Python users. The MATLAB version offers GUI and code scripts. The Python version uses NumPy for simple code development, and PyTorch for high computation performance. And it provides a step-by-step guide in terminal command. pNet provides streamlined workflow to carry out computation and visualization of pFNs. </p>
            <p>It also integrates several statistical methods to investigate the relationship between pFNs and behavior data. In addition, quality control is available to check the quality of pFN modeling results. This toolbox can be downloaded from <a href="https://github.com/YuncongMa/pNet">YuncongMa/pNet</a> and <a href="https://github.com/MLDataAnalytics/pNet">MLDataAnalytics/pNet</a>.</p>
            <img src="/images/Components/Machine_Learning_Models/pNet/Picture1.jpg" alt=""/>
            <i>Figure 7. pNet network model</i>  
          </div>
         
        </div>
        
      </>
    ),
    'Harmonization': (
      <>
      <div className={styles.section} id="combat_overview">
        <div className={styles.headerFlex}>
          <img className={styles.brainImage} src="/images/Components/brain_harmonize.png"/>
          <h1 style={{ color: '#92da44' }}>NiChart Data Harmonization</h1>
        </div>
        <p>To estimate and remove scanner-related batch effects in imaging variables we apply a statistical harmonization method, <a href="https://github.com/PennSIVE/ComBatFam_Pipeline/tree/main/ComBatFamily">ComBat</a>. The ComBat method is a Bayesian statistical technique aimed at removing <em>batch effects</em> in high-dimensional datasets.</p>
        <p>The method estimates both the mean (<em>location</em>) and the variance (<em>scale</em>) of the residuals across batches using <em>Empirical Bayes</em> estimation, after correcting for additional covariates, such as age, sex and ICV.</p>

        <br></br><br></br><hr/><br></br><br></br>

        <div className={styles.subsection} id="combat_family">
          <h2>Combat Family of Statistical Harmonization Tools</h2>
          <p>NiChart data harmonization will be powered by the Combat-family software package that provides an ensemble of harmonization tools. Variants like ComBat-GAM offer the possibility to model selected covariates using splines, providing flexible adjustments to non-linear covariate associations. Combat can be used through a train/test paradigm, applying it on a training set to estimate batch effect parameters, and using the existing model to harmonize new data from the same batches.</p>
          <img src="/images/Components/Data_Harmonization/Picture1b.png" alt=""/>
          <i>Figure 1. Combat family of tools</i>
        </div>

        <br></br><br></br><hr/><br></br><br></br>

        <div className={styles.subsection} id="combat_tools">
          <h2>Combat Visualization and QC</h2>
          <p>Combat visualization and quality control (QC) package provides tools for evaluating batch effects and estimated parameters before and/or after harmonization.</p>
          <div className={styles.harmonizationPictures}>
            <img src="/images/Components/Data_Harmonization/Picture4.png" alt=""/>
            <img src="/images/Components/Data_Harmonization/Picture5.png" alt=""/>
            <img src="/images/Components/Data_Harmonization/Picture6.png" alt=""/>
            <img src="/images/Components/Data_Harmonization/Picture3.png" alt=""/>
          </div>
          <i>Figure 2. Combat visualization and QC tool functions</i>
        </div>
      </div>
      </>
    ),
    'Machine Learning Models': (
      <>
        <div className={styles.section} id="ml_overview">
          <div className={styles.headerFlex}>
            <img className={styles.brainImage} src="/images/Components/brain_learn.png"/>      
            <h1 style={{ color: '#e9a944' }}>NiChart ML Models</h1>
          </div>
          <p>NiChart offers an extensible library of pre-trained machine learning (ML) models that can convert high-dimensional imaging data into low-dimensional imaging signatures. These representations effectively capture and quantify brain changes associated with specific diseases or neurodegenerative conditions.</p>
          <p>The collection of NiChart imaging signatures contributes to the neuroimaging chart dimensional system. NiChart's pre-trained ML models are <a href="https://hub.docker.com/r/cbica/nichart-sparescores">readily available</a>, thereby eliminating the need for extensive training or expertise in machine learning. Additionally, the extensibility of the NiChart library will allow researchers to add their own specialized models, after harmonizing their data with NiChart.</p>
          <p>The models are trained on carefully selected subsets of the reference dataset, tailored to each task and target disease/condition.</p>
        
          
          <br></br><br></br><hr/><br></br><br></br>

          <div className={styles.subsection} id="ml_supervised">
            <h2>Supervised models</h2>
            <h3>SPARE Models</h3>
            <p><a href="https://github.com/CBICA/spare_score">SPARE</a> or Spatial Patterns of Abnormality for Recognition of Disease models are predictive supervised learning methods that have been extensively validated. SPARE models train on imaging features extracted from single or multi-modal MRI scans. The models use these features to learn how to identify patterns in the brain that are associated with different diseases. Initial models are provided for SPARE-BA (brain age) and SPARE-AD (Alzheimer's disease). Additional models for SPARE-CVD (cardio-vascular disease risk), SPARE-DM (Type2 diabetes), SPARE-SCZ (schizophrenia) and SPARE-CD (chronic depression) as well as models derived from weakly supervised methods and which identify subtypes of these diseases, will be added in future releases.</p>
            <img src="/images/Components/Machine_Learning_Models/aibil/sparead_frombrainpaper.gif" alt=""/>
            <i>Figure 1. Grey matter and white matter group differences between individuals with low vs high SPARE-AD values (from <a href="https://academic.oup.com/brain/article/132/8/2026/266984">1</a>).</i>
          </div>

          <br></br><br></br><hr/><br></br><br></br>

          <div className={styles.subsection} id="ml_semisupervised">
            <h2>Weakly-supervised models</h2>
            <h3>Image-based Disease Heterogeneity Models</h3>
            <p>Our research team has developed ML tools to uncover imaging patterns of disease heterogeneity from MRI data. These tools help us identify distinct disease subtypes that shed light on the underlying neuroanatomical differences associated with various pathologies. Our previous work has identified four distinct disease subtypes for Alzheimer's disease and two subtypes for schizophrenia. The pre-trained models provided in NiChart will enable users to obtain more nuanced measures beyond the traditional disease scores.</p>
            <img src="/images/Components/Machine_Learning_Models/aibil/smilegan_naturefig.png" alt=""/>
            <i>Figure 2. Alzheimer's disease subtypes identified by the SMILE-GAN method (from <a href="https://www.nature.com/articles/s41467-021-26703-z">2</a>).</i>
          </div>  
        </div>
        
      </>
    ),
    'Data Visualization': (
      <>
        <div className={styles.section} id="datavis_overview">
          <div className={styles.headerFlex}>
            <img className={styles.brainImage} src="/images/Components/brain_visualize.png"/>
            <h1 style={{ color: '#f5e852' }}>NiChart Data Visualization</h1>
          </div>
          <p>NiChart's visualization modules offer tools to assist users in comparing outcome variables extracted from their MRI data against established NiChart reference distributions. Users can effectively visualize and interpret their data, gaining meaningful insights into their individual profiles.</p>
          <p>NiChart offers two convenient options for visualizing user data: A client-side visualization tool integrated with the cloud portal enables users to derive and visualize NiChart dimensions for their data on the browser. A PyQT-based installable package provides extended capabilities for exploring and analyzing user data.</p>
          
          <br></br><br></br><hr/><br></br><br></br>

          <div className={styles.subsection} id="datavis_viewer">
            <h2>NiChart Viewer</h2>
            <p>Alternatively, users can install the <a href="https://github.com/CBICA/niCHART">NiChart viewer</a>, a PyQT-based package that provides an extended set of visualization functionality. </p>
            <img src="/images/Components/DataViewers/nichart_viewer.png" alt=""/>
            <i>Figure 2. NiChart Viewer.</i>
          </div>

          <br></br><br></br><hr/><br></br><br></br>

          <div className={styles.subsection} id="datavis_webviewer">          
            <h2>NiChart Web Viewer</h2>
            <p>We also provide a web viewer that is integrated with our <a href="/portal">cloud portal</a> to provide a more practical option for visualization of derived imaging features and final signatures.The viewer is designed as an in-browser application to provide very fast rendering of visualizations. The application provides options to select the target variable for visualization, and the reference data used as the comparison set. This will allow users to compare their selected data to different disease or demographic groups directly from the cloud portal. </p>

            <img src="/images/Components/DataViewers/nichart_webviewer_plots.png" alt=""/>
            <h3>Coming Soon: Scan Visualization</h3>
            <p>In a near future (Q1 2024) release, we will also provide in-browser viewing of scans with region-of-interest overlays, directly from the visualization page. This feature will allow users to view scans and their selected region of interest highlighted with just a click.</p>
            <img src="/images/Components/DataViewers/nichart_webviewer_mri.png" alt=""/>
          </div>
        </div>
      </>
    ),
    'Deployment': (
      <>
        <div className={styles.section} id="deploy_overview">      
          <div className={styles.headerFlex}>
            <img className={styles.brainImage} src="/images/Components/brain_deploy.png"/>
            <h1 style={{ color: '#ac29d8' }}>NiChart Software Deployment and Application</h1>
          </div>
          <p>NiChart provides three installation options to accommodate a wide range of end-users: <a href="https://github.com/CBICA/NiChart_Project">local</a> installation, <a href="https://hub.docker.com/repository/docker/aidinisg/nichart_dlmuse/general">containerized</a> installation, and the <a href="/portal">cloud portal</a>.</p>
          <p>The choice of installation option depends on the user's technical expertise, computational resources, and desired level of control. For users with strong technical skills and a need for maximum flexibility, local user-managed or containerized installation is recommended. For users who require a highly accessible and user-friendly solution, the cloud portal is the ideal choice.</p>
          <p>Currently, the portal provides a restricted pipeline that is limited to structural MRI images.</p>

          <br></br><br></br><hr/><br></br><br></br>

          <div className={styles.subsection} id="deploy_install">
            <h2>Open-source Software Packages</h2>
            <p>NiChart is designed with a modular architecture, consisting of independent software components that can be installed and applied individually. This modular approach was chosen to ensure the extensibility of NiChart in the future without creating a dependency nightmare. Users can easily download these components followink the links at the <a href="https://github.com/CBICA/NiChart_Project">NiChart_Project Github</a> page. The installation process typically involves downloading the component, extracting the files, and running a setup script.</p>
          </div>
          
          <br></br><br></br><hr/><br></br><br></br>
          
          <div className={styles.subsection} id="deploy_container">
            <h2>Docker and Singularity Containers</h2>
            <p>We use the power of containerization technology for major image processing modules to simplify complex workflows and to ensure compatibility across different computing environments.</p>
          </div>
          
          <br></br><br></br><hr/><br></br><br></br>
          
          <div className={styles.subsection} id="deploy_cloud">
            <h2>NiChart Cloud Portal</h2>
            <p>The NiChart cloud portal is a user-friendly online platform that streamlines the process of analyzing structural magnetic resonance imaging (sMRI). It provides a straightforward interface that allows users to upload their sMRI images, apply pre-trained ML models to extract meaningful biomarkers, and visualize the results in an intuitive manner.</p>
          </div>
        </div>
      </>
    ),
    
  };

  return (
    <div className={styles.container}>
      <Head>
        <title>NiChart | Components</title>
        <Favicons />
      </Head>
      <Header />
      <div className={styles.componentsPage}>
        <Sidebar currentSection={expandedSection} updateExpandedSection={setExpandedSection}/>
        <div>
          <div className={styles.componentsContainer}>
            {contentBySection[expandedSection]}
          </div>
        </div>
      </div>
      <Footer />
    </div>
  );
};

export default Components;
