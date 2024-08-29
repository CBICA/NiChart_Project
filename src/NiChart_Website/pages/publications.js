import React from 'react';
import Head from 'next/head';
import Header from '../components/Layout/Header';
import Footer from '../components/Layout/Footer';
import styles from '../styles/Publications.module.css';
import Favicons from '../components/Favicons/Favicons';

const Publications = () => {
    return (
      <div className={styles.container}>
        <Head>
          <title>NiChart | Publications</title>
          <Favicons />
        </Head>
        <Header />
        <main className={styles.mainContent}>
          <h1 className={styles.title}>Publications</h1>
          <p className={styles.description}>
          NiChart is built upon the foundation of extensive research conducted by a diverse group of contributors. This page highlights a selection of published papers that have directly influenced or been utilized within NiChart components. This list will be continuously updated to reflect the latest NiChart-related publications.</p>
          <section className={styles.publicationSection}>
            <h2 className={styles.sectionHeader}>sMRI Image Processing</h2>
              <ul>
                <li>Doshi J, Erus G, Habes M, Davatzikos C. DeepMRSeg: A convolutional deep neural network for anatomy and abnormality segmentation on MR images. ArXiv Prepr ArXiv190702110. Published online 2019.</li>
                <li>Doshi J, Erus G, Ou Y, et al. MUSE: MUlti-atlas region Segmentation utilizing Ensembles of registration algorithms and parameters, and locally optimal atlas selection. NeuroImage. 2016;127:186–195.</li>
                <li>Wen, J., Nasrallah, I. M., Abdulkadir, A., Satterthwaite, T. D., Yang, Z., Erus, G., Singh, A., Sotiras, A., Mamourian, E., Doshi, J., Cui, Y., Srinivasan, D., Skampardoni, I., Chen, J., Hwang, G., Bergman, M., Bao, J., Veturi, Y., Zhou, Z., Davatzikos, C. (2023). Genomic loci influence patterns of structural covariance in the human brain. Proceedings of the National Academy of Sciences, 120(52), e2300842120. https://doi.org/10.1073/pnas.2300842120</li>
                <li>Ou, Y., Sotiras, A., Paragios, N. & Davatzikos, C. DRAMMS: Deformable registration via attribute matching and mutual-saliency weighting. Med Image Anal 15, 622-639.</li>
              </ul>
          <h2 className={styles.sectionHeader}>fMRI and DTI Image Processing</h2>
            <ul>
              <li>Cieslak, M., Cook, P.A., He, X. et al. QSIPrep: an integrative platform for preprocessing and reconstructing diffusion MRI data. Nat Methods 18, 775–778 (2021). https://doi.org/10.1038/s41592-021-01185-5</li>
              <li>Esteban, O., Markiewicz, C.J., Blair, R.W. et al. fMRIPrep: a robust preprocessing pipeline for functional MRI. Nat Methods 16, 111–116 (2019). https://doi.org/10.1038/s41592-018-0235-4</li>
              <li>Mehta K, Salo T, Madison T, Adebimpe A, Bassett DS, Bertolero M, Cieslak M, Covitz S, Houghton A, Keller AS, Luo A, Miranda-Dominguez O, Nelson SM, Shafiei G, Shanmugan S, Shinohara RT, Sydnor VJ, Feczko E, Fair DA, Satterthwaite TD. XCP-D: A Robust Pipeline for the post-processing of fMRI data. bioRxiv [Preprint]. 2023 Nov 21:2023.11.20.567926. doi: 10.1101/2023.11.20.567926. PMID: 38045258; PMCID: PMC10690221.</li>
            </ul>

          <h2 className={styles.sectionHeader}>Harmonization</h2>
            <ul>
              <li>Fortin J-P, Cullen N, Sheline YI, et al. Harmonization of cortical thickness measurements across scanners and sites. NeuroImage. 2018;167:104–120. doi:10.1016/j.neuroimage.2017.11.024</li>
              <li>Pomponio R, Erus G, Habes M, et al. Harmonization of large MRI datasets for the analysis of brain imaging patterns throughout the lifespan. NeuroImage. 2020;208:116450.</li>
            </ul>
            
          <h2 className={styles.sectionHeader}>MRI Analysis</h2>
            <ul>
              <li>Da X, Toledo JB, Zee J, et al. Integration and relative value of biomarkers for prediction of MCI to AD progression: Spatial patterns of brain atrophy, cognitive scores, APOE genotype and CSF biomarkers. NeuroImage Clin. 2014;4(0):164–173.</li>
              <li>Davatzikos C, Xu F, An Y, Fan Y, Resnick SM. Longitudinal progression of Alzheimer’s-like patterns of atrophy in normal older adults: the SPARE-AD index. Brain. 2009;132(8):2026–2035.</li>
            </ul>

          <h2 className={styles.sectionHeader}>Supervised Machine Learning</h2>
            <ul>
              <li>Bashyam VM, et al. MRI signatures of brain age and disease over the lifespan based on a deep brain network and 14 468 individuals worldwide. Brain. 2020 Jul 1;143(7):2312-2324.</li>
              <li>Davatzikos, C. Machine learning in neuroimaging: Progress and challenges. Neuroimage 197, 652-656, doi:10.1016/j.neuroimage.2018.10.003 (2019).</li>
              <li>Habes, M. et al. The Brain Chart of Aging: Machine-learning analytics reveals links between brain aging, white matter disease, amyloid burden, and cognition in the iSTAGING consortium of 10,216 harmonized MR scans. Alzheimers Dement 17, 89-102 (2021).</li>
              <li>Habes M, Janowitz D, Erus G, et al. Advanced Brain Aging: relationship with epidemiologic and genetic risk factors, and overlap with Alzheimer disease atrophy patterns. Transl Psychiatry. 2016;6:e775.</li>
            </ul>

          <h2 className={styles.sectionHeader}>Semi-supervised Machine Learning</h2>
            <ul>
              <li>Chand, G., et al. Two Distinct Neuroanatomical Subtypes of Schizophrenia Revealed Using Machine Learning. Oxford Press, 2020.</li>
              <li>Dong, A., et al. CHIMERA: Clustering of heterogeneous disease effects via distribution matching of imaging patterns. IEEE Trans Med Imaging, 2016. 35(2): p. 612-621.</li>
              <li>Dong, A. et al. Heterogeneity of neuroanatomical patterns in prodromal Alzheimer's disease: links to cognition, progression and biomarkers. Brain 140, 735-747 (2017).</li>
              <li>Eavani H, Habes M, Satterthwaite TD, et al. Heterogeneity of structural and functional imaging patterns of advanced brain aging revealed via machine learning methods. Neurobiol Aging. 2018;71:41–50.</li>
              <li>Habes M, Grothe MJ, Tunc B, McMillan C, Wolk DA, Davatzikos C. Disentangling Heterogeneity in Alzheimer’s Disease and Related Dementias Using Data-Driven Methods. Biol Psychiatry. Published online January 31, 2020.</li>
              <li>Varol, E., et al., HYDRA: Revealing heterogeneity of imaging and genetic patterns through a multiple max-margin discriminative analysis framework. Neuroimage, 2017. 145(Pt B): p. 346-364.</li>
              <li>Wen, J. et al. Characterizing Heterogeneity in Neuroimaging, Cognition, Clinical Symptoms, and Genetics Among Patients With Late-Life Depression. JAMA Psychiatry 79, 464-474 (2022).</li>
              <li>Wen, J. et al. Multi-scale semi-supervised clustering of brain images: Deriving disease subtypes. Med Image Anal 75, 102304 (2022).</li>
              <li>Yang, Z. et al. A deep learning framework identifies dimensional representations of Alzheimer’s Disease from brain structure. Nature Communications 12 (2021).</li>
              <li>Yang, Z., Wen, J. & Davatzikos, C. Surreal-GAN:Semi-Supervised Representation Learning via GAN for uncovering heterogeneous disease-related imaging patterns. International Conference on Learning Representations (ICLR) (2022).</li>
            </ul>
          </section>
        </main>
        <Footer />
      </div>
    );
};

export default Publications;
