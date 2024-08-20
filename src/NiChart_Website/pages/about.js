import React from 'react';
import Head from 'next/head';
import Header from '../components/Layout/Header';
import Footer from '../components/Layout/Footer';
import Favicons from '../components/Favicons/Favicons';
import styles from '../styles/About.module.css';
import { Grid, Typography, Paper } from '@mui/material';
import Link from 'next/link';

const About = () => {
  return (
    <div className={styles.container}>
      <Head>
        <title>NiChart | About</title>
        <Favicons />
      </Head>
      <Header />
      <div className={styles.mainContent}>  
        <Grid container rowSpacing={{xs: 3, sm: 4, md: 20}} columnSpacing={{ xs: 1, sm: 2, md: 3 }}>
          <Grid item xs={12} md={6}>
            <Paper className={styles.paper} square={false}>
              <Typography variant="h2" gutterBottom>About NiChart</Typography>
              <Typography variant="body1">
                NiChart is a set of modular but integrated software tools for neuroimaging research, and a cloud-based web application to provide wide access to these tools.
                This work was supported, in part, by NIH grants U24NS130411 and RF1AG054409. The Cloud implementation was also supported by Amazon Web Services (AWS).
                <br></br>
              </Typography>
              <Typography variant="h2" gutterBottom>Applications</Typography>
              <Typography variant="body1">NiChart enables mapping of large-scale multi-modal brain MRI data into a dimensional system of neuroimaging derived measures, including signatures implemented by machine learning (ML) models. </Typography>
              <ul>
                <li>
                  <Typography variant="body1">We provide image processing tools for deriving a panel of imaging derived phenotypes from structural (sMRI), diffusion (dMRI) and functional (fMRI) imaging data: from ROIs and functional networks to structural covariance and ML indices.</Typography>
                </li>
                <li>
                  <Typography variant="body1">ML models, which are previously trained on  pre-processed, QC'ed and curated reference samples, allow users to calculate imaging signatures that quantify complex multi-variate imaging patterns of brain changes mapping the image data into a small but informative set of neuroimaging chart dimensions. </Typography>
                </li>
                <li>
                  <Typography variant="body1">ML models capture heterogeneity of brain aging and neurodegeneration, and atrophy patterns due to various diseases and conditions, such as Alzheimer's disease, neuropsychiatic disorders, or cardio-vascular risk factors. </Typography>
                </li>
                <li>
                  <Typography variant="body1">Standardized values from the large reference set allow users to compare their data with NiChart-based normative ranges or distributions from specific disease subgroups.</Typography>
                </li>
              </ul>
            </Paper>
          </Grid>
          <Grid item xs={12} md={6}>
            <div className={styles.imageContainer}>
              <img src="/images/About/nichart_logo_anim_v22_C_ms150.gif" alt="" className={styles.image} />
            </div>
            </Grid>
          <Grid item xs={12} md={6}>
            <Paper className={styles.paper} square={false}>
              <Typography variant="h4" gutterBottom>FDA Disclaimer</Typography>
              <Typography variant="body1">Please be advised that NiChart is a set of free software tools provided for research purposes. The statements made regarding the products have not been evaluated by the Food and Drug Administration. The efficacy of these products has not been confirmed by FDA-approved research. These products are not intended for clinical purposes. All information presented here is not meant as a substitute for or alternative to information from health care practitioners.</Typography>
            </Paper>
          </Grid>
          <Grid item xs={12} md={6}>
            <Paper className={styles.paper} square={false}>
              <Typography variant="h4" gutterBottom>NiChart Cloud Privacy Statement</Typography>
              <Typography variant="body1">For convenience, NiChart is offered as a web service via NiChart Cloud, a service hosted using Amazon Web Services infrastructure. By uploading your data to NiChart, you are agreeing that you have valid, authorized access to that data and are not uploading personally-identifiable health information as defined by HIPAA. Uploaded scans are placed in a secure backend storage location in a private segment of the Amazon infrastructure  accessible via your login account. Individuals having root access to this server could also access your data for system maintenance purposes (e.g. to occasionally monitor folder size and delete data). Other than for system maintenance operations,  individuals with root access will never access, use or share your data with anyone.  Any uploaded data is retained for a maximum of 36 hours before being deleted. You may also choose to delete data immediately from the NiChart Cloud interface. By choosing to use NiChart Cloud, you agree that you understand these terms. If you wish to revoke this agreement at any time, simply discontinue using the service. You may contact us directly at <b><a href="mailto:nichart-devs@cbica.upenn.edu">nichart-devs@cbica.upenn.edu</a></b> about concerns related to the handling of your data. At your preference, you may also download the NiChart software tools for use on your own machine. Please see the components page for details.</Typography>
            </Paper>
          </Grid>
        </Grid>
      </div>
      <Footer />
    </div>
  );
};

export default About;
