import React from 'react';
import Head from 'next/head';
import Header from '../components/Layout/Header';
import Footer from '../components/Layout/Footer';
import Favicons from '../components/Favicons/Favicons';
import styles from '../styles/FAQ.module.css';
import Link from 'next/link';
import { Grid, Paper, Box, Typography, Accordion, AccordionSummary, AccordionDetails } from '@mui/material';
import { Heading, Divider } from '@aws-amplify/ui-react';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';


const FAQ = () => {
  return (
    <div className={styles.container}>
      <Head>
        <title>NiChart | FAQ</title>
        <Favicons />
      </Head>
      <Header />
      <div className={styles.mainContent}>
        <Paper className={styles.paper}>
          <Box textAlign="center">
            <Heading level={1}>Frequently Asked Questions</Heading>
            <Divider/>
          </Box>
          <Box>
            <p></p>
            <p></p>
            <br></br>

            <Accordion>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography variant="h6">What is NiChart?</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <p>NiChart is a platform designed to interpret and visualize brain imaging data, providing tools and resources for professionals and researchers in the field of neuroimaging.</p>
              </AccordionDetails>
            </Accordion>

            <Accordion>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography variant="h6">Who can benefit from using NiChart?</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <p>NiChart is beneficial for neuroscientists, researchers, medical professionals, and anyone involved in the analysis and visualization of brain imaging data.</p>
              </AccordionDetails>
            </Accordion>

            <Accordion>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography variant="h6">Is there a quickstart guide for new users?</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <p>Yes, new users can refer to the <a href="/quickstart/">Quickstart</a> page which provides a step-by-step guide on how to get started with the platform's tools and features.</p>
              </AccordionDetails>
            </Accordion>
            
            <br></br><br></br>
            <Divider/>
            <br></br><br></br>

            <Accordion>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography variant="h6">What types of data does NiChart support?</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <p>At this moment, the <a href="/portal/">NiChart Cloud</a> portal has pipelines that support structural MRI scans (in the .nii.gz format), but the <a href="https://github.com/CBICA/NiChart_Project">NiChart_Project</a> will be continuously updated with pipelines that support fMRI and DTI scans as well. Please refer to the <a href="/components/#Image%20Processing">Components</a> page for more information on the components that will be integrated to the platform.</p>
                <p>Alternatively, you can use the image processing components that will soon be integrated already in their local versions (see the <a href="/components/#Image%20Processing">Components</a> page for more info).</p>
              </AccordionDetails>
            </Accordion>

            <Accordion>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography variant="h6">Can I use my data with NiChart?</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <p>Of course! Be sure to carefully read the <a rel="noopener noreferrer" href="https://www.upenn.edu/about/privacy-policy" className={styles.cbicaLink}>Privacy Policy</a> and the <a target="_blank" rel="noopener noreferrer" href="https://www.upenn.edu/about/disclaimer" className={styles.cbicaLink}>UPenn Disclaimer</a> before uploading any data.</p>
                <p>By using NiChart Cloud, you agree to share your uploaded image data with the University of Pennsylvania for processing only. Please see the <a href="/about/">About</a> page for more details. All data is deleted after a maximum of 36 hours.</p>

              </AccordionDetails>
            </Accordion>

            <Accordion>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography variant="h6"> Is there customer support for technical issues on the platform?</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <p>You may contact us directly at <b><a href="mailto:nichart-devs@cbica.upenn.edu">nichart-devs@cbica.upenn.edu</a></b> about any problems, questions, feedback or concerns related to the handling of your data that arise from the usage of the NiChart platform.</p>
              </AccordionDetails>
            </Accordion>

            <Accordion>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography variant="h6">Where can I find the pipelines / models / data in order to perform the NiChart Cloud tasks locally?</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <p>It's our goal for you to be able to run the same cloud pipelines locally on your infrastructure. Below, you can find a list of packages, along with the respective models and data, required to execute these pipelines. Please note that some packages are computationally intensive and thus benefit from a capable system, ideally with GPU support.</p>
                <br></br>
                <p><b>Image Processing:</b></p>
                <p>You can access the image processing pipeline we utilize in the <a href="https://github.com/CBICA/NiChart_DLMUSE">NiChart_DLMUSE</a> package. It integrates components from both the <a href="https://github.com/CBICA/DLICV">DLICV</a> and <a href="https://github.com/CBICA/DLMUSE">DLMUSE</a> pipelines, allowing you the option to run specific segments individually. This pipeline is built on the <a href="https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1">nnUNet</a> framework, and detailed instructions are available in the repository's <a href="https://github.com/CBICA/NiChart_DLMUSE/blob/main/README.md">Readme</a>. Models for all pipelines are located in the <a href="https://github.com/CBICA/NiChart_DLMUSE/releases">Releases</a> section as release <a href="https://github.com/CBICA/NiChart_DLMUSE/releases/tag/0.1.7">assets</a>. All pipelines are accompanied with a docker container, which you can easily download and run without any installation procedure. More information can be found in the repositories' installation and usage instructions.</p>
                <p>More pipelines are scheduled to be integrated in the NiChart Cloud, and of course made available through public repositories for local execution.</p>
                <br></br>
                <p><b>Machine Learning:</b></p>
                <p>For those interested in leveraging our Machine Learning pipelines, we provide the <a href="https://github.com/CBICA/spare_score">spare_score</a> pipeline. The necessary models to operate the SPARE_scores pipeline are conveniently located in the repository's <a href="https://github.com/CBICA/spare_score/tree/main/spare_scores/mdl">/spare_scores/mdl/</a> directory. Alternatively, you can find the pipeline as a <a href="https://hub.docker.com/repository/docker/aidinisg/spare_scores/general">docker container</a> (models included).</p>
                <p>Machine learning pipelines are models are constantly under development and will be integrated to the NiChart Cloud as soon as possible and we will be providing public repositories for local execution.</p>
                <br></br>
                <p><b>Visualization:</b></p>
                <p>While we recommend visualizing the results of your pipeline through the NiChart Cloud, we are commited to providing alternatives for local execution and visualization. To that end, we have an open-source, PyQT-based alternative called <a href="https://github.com/gurayerus/NiChart_Viewer">NiChart_Viewer</a>.</p>
                <br></br>
                <p><b>Data:</b></p>
                <p>The data we've used for model training, reference centile curves, testing and validation come from the <a href="https://www.med.upenn.edu/cbica/imaging-consortia-increasing-sample-size-and-understanding-heterogeneity-in-health-and-disease.html">iSTAGING</a> project.</p>
                <p>For more information, or inquiries about the data, please contact us at <a href="mailto:nichart-devs@cbica.upenn.edu">nichart-devs@cbica.upenn.edu</a>.</p>
                <p>We will be happy to help you and answer any questions that you might have!</p>
              </AccordionDetails>
            </Accordion>

            <br></br><br></br>
            <Divider/>
            <br></br><br></br>

            <Accordion>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography variant="h6">Can I request new features or suggest improvements for NiChart?</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <p>User feedback is valuable, and suggestions for new features or improvements can be submitted through the website's <a href="/feedback/">Feedback</a> section.</p>
              </AccordionDetails>
            </Accordion>

            <br></br><br></br>
            <b>We will update this FAQ as we receive more feedback. Stay tuned!</b>
          </Box>

        </Paper>
      </div>
      <Footer />
    </div>
  );
};

export default FAQ;
