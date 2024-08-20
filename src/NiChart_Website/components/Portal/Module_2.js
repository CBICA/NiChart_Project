import { React, useState } from 'react';
import { Flex, Heading, Divider} from '@aws-amplify/ui-react';
import { SpareScoresInputStorageManager, SpareScoresDemographicStorageManager, JobList, launchSpareScores, getSpareScoresOutput, emptyBucketForUser, uploadToModule2, getCombinedCSV, downloadBlob } from '../../utils/uploadFiles.js'
import { getUseModule1Results, setUseModule1Results, setUseModule2Results, getModule2Cache, getModule1Cache } from '../../utils/NiChartPortalCache.js'
import styles from '../../styles/Portal_Module_2.module.css'
import { ResponsiveButton as Button } from '../Components/ResponsiveButton.js'
import Modal from '../Components/Modal';
import { ModelSelectionMenu } from './ModelSelectionMenu.js'

async function exportModule2Results(moduleSelector) {
    // Perform the caching transfer operation
    setUseModule2Results(true);
    await getSpareScoresOutput(false);
    let cachedResult = await getModule2Cache();
    if (Object.keys(cachedResult).length === 0) {
        alert("We couldn't export your results because there doesn't appear to be output from Module 2. Please generate the output first or upload the file to Module 3 manually.")
        return;
    }
    // Switch to module 3
    moduleSelector("module3");
}

function Module_2({moduleSelector}) {
  const [useModule1Cache, setUseModule1Cache] = useState(getUseModule1Results());
  
  // Modal dialog stuff for model selection
  const [modelSelectionModalOpen, setModelSelectionModalOpen] = useState(false);
  const handleModelSelectionOpen = () => setModelSelectionModalOpen(true);
  const handleModelSelectionClose = () => setModelSelectionModalOpen(false);
  
  async function disableModule1Results() {
      setUseModule1Results(false);
      setUseModule1Cache(false);
  }
  
  async function enableModule1Results() {
      setUseModule1Results(true);
      setUseModule1Cache(true);
      await getCombinedCSV(false);
      let cachedResult = await getModule1Cache();
      if (Object.keys(cachedResult).length === 0) {
         alert("We couldn't import your results because there doesn't appear to be output from Module 1. Please generate the output first or upload the file to Module 2 manually.")
         return;
      }
      await uploadToModule2(cachedResult.csv)
      

  }
  
  async function downloadTemplateDemographics() {
    const referenceFilePath = '/content/Portal/Module2/TemplateDemographicsCSV.csv'
    try {
        const response = await fetch(referenceFilePath);
        if (response.status === 404) {
            console.error('Error loading template CSV:', response.statusText);
            alert("We couldn't download the template CSV. Please submit a bug report.")
            return;
        }
        const content = await response.text();
        //referenceData = Papa.parse(content, { header: true }).data;
        
        const fileName = "Template-Demographics-CSV.csv";
        const fileType = "text/csv";
        const blob = new Blob([content], { type: fileType });
        
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = fileName || 'download';
        const clickHandler = () => {
            setTimeout(() => {
            URL.revokeObjectURL(url);
            a.removeEventListener('click', clickHandler);
            }, 150);
        };
        a.addEventListener('click', clickHandler, false);
        a.click();
        
    } catch (error) {
        console.error('Error loading template CSV:', error);
        alert("We couldn't download the template CSV. Please check your connection or submit a bug report.")
        return;
    }


  }
    
  return (
    <div>
      <Heading level={1}>Module 2: Machine Learning</Heading>
      <div className={styles.moduleContainer}>
          <Divider orientation="horizontal" />
          <Flex direction={{ base: 'column', large: 'row' }} maxWidth="100%" padding="1rem" width="100%" justifyContent="flex-start">
              <Flex justifyContent="space-between" direction="column" width="33%">
                <Heading level={3}>Upload Subject CSV</Heading>
                Upload your ROI volume CSV. Alternatively, import your results directly from Module 1. If you clear your output data from Module 1, you may need to import it again for your jobs in this module to succeed.
                { !getUseModule1Results() && (<SpareScoresInputStorageManager />)}
                { !getUseModule1Results() && (<Button variation="primary" colorTheme="info" loadingText="Importing..." onClick={async () => await enableModule1Results()}>Import from Module 1</Button>)}
                { getUseModule1Results() && (<p>Using results from Module 1!</p>)}
                { getUseModule1Results() && (<Button variation="primary" colorTheme="info" onClick={async () => await disableModule1Results()}>Upload a CSV Instead</Button>) }
                <Heading level={3}>Upload Demographic CSV</Heading>
                <p>This file should correspond to the scans present in the ROI CSV, and should contain demographic data. Scans should be on individual rows and IDs should correspond to the original T1 filename (without the extension). At minimum, your file should have columns for ID, Age (in years) and Sex (M or F).</p>
                <p>You may download an example template for this file with the "Download Template" button.</p>
                <SpareScoresDemographicStorageManager />
                <Button loadingText="Selecting..." variation="primary" onClick={handleModelSelectionOpen}>Select Models</Button> 
                <Modal
                    open={modelSelectionModalOpen}
                    handleClose={handleModelSelectionClose}
                    title="Select SPARE models"
                    content="Check any number of models to use during SPARE score generation. This list will be expanded as we release new models."
                >
                    <ModelSelectionMenu />
                </Modal>
                <Button loadingText="Submitting..." variation="primary" onClick={async () => launchSpareScores() } >Generate SPARE scores</Button>
                <Button loadingText="Downloading..." variation="primary" colorTheme="info" onClick={async () => downloadTemplateDemographics() }>Download Template</Button>
              </Flex>
              <Divider orientation="vertical" />
              <Flex direction="column" width="33%">
                <Heading level={3}>Jobs in Progress</Heading>
                SPARE scores that are currently being calculated will appear here. Finished jobs will be marked with green. Please wait for your jobs to finish before proceeding. If your job fails, please contact us and provide the job ID listed below.
                <p>Jobs should reach the RUNNING phase in under a minute and complete within 30 seconds.</p>
                <JobList jobQueue="cbica-nichart-sparescores-jobqueue" />
              </Flex>
              <Divider orientation="vertical" />
              <Flex direction="column" width="33%">
                <Heading level={3}>Download SPARE Output</Heading>
                Here you can download the results (a merged CSV with ROI volumes, demographic info, and calculated SPARE scores for each scan).
                You can also export this file directly to module 3 for visualization.
                <Button loadingText="Downloading CSV..." variation="primary" onClick={async () => getSpareScoresOutput(true) } >Download SPARE score CSV</Button>
                <Button loadingText="Exporting..." variation="primary" colorTheme="info" onClick={async () => exportModule2Results(moduleSelector) } >Export to Module 3: Visualization</Button>
                <Button loadingText="Emptying..." variation="destructive" onClick={async () => emptyBucketForUser('cbica-nichart-outputdata', 'sparescores/')} >Clear All Data</Button>
              </Flex>
          </Flex>
      </div>
    </div>
  );
}

export default Module_2;