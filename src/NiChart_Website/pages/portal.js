import React, { useState } from 'react';
import Head from 'next/head';
import Header from '../components/Layout/Header';
import Footer from '../components/Layout/Footer';
import Favicons from '../components/Favicons/Favicons';
import Sidebar from '../components/Portal/Sidebar';
import Module_1 from '../components/Portal/Module_1';
import Module_2 from '../components/Portal/Module_2';
import Module_3 from '../components/Portal/Module_3';
import NiChartAuthenticator from '../components/Portal/NiChartAuthenticator';
import styles from '../styles/Portal.module.css'
import { Heading, Button, Flex, Authenticator } from '@aws-amplify/ui-react';
import awsExports from '../utils/aws-exports';

function Portal() {

  // State to track the selected module
  const [selectedModule, setSelectedModule] = useState('module1'); 
  // Function to handle module selection
  const handleModuleSelection = (module) => {
    setSelectedModule(module);
  };
  
  return (
  <div>
  <NiChartAuthenticator>
  {({ signOut, user }) => (
    <div className={styles.container}>
      <Head>
        <title>NiChart | Portal</title>
        <Favicons />
      </Head>
      <Header user={user} signOut={signOut}/>
      <div className={styles.portalPage}>
        <Sidebar handleModuleSelection={handleModuleSelection}/>
        <div className={styles.modulePage}>
            {selectedModule === 'module1' && <Module_1 moduleSelector={handleModuleSelection} />}
            {selectedModule === 'module2' && <Module_2 moduleSelector={handleModuleSelection} />}
            {selectedModule === 'module3' && <Module_3 moduleSelector={handleModuleSelection} />}
            <div>
              <h4> By using NiChart Cloud, you agree to share your uploaded image data with the University of Pennsylvania for processing only. Please see the <a href="/about" >About page</a> for more details. All data is deleted after a maximum of 36 hours. </h4>
              To use, drop files into the box on the left. When results are available, click to download.
              Jobs may take up to 6 minutes to start depending on resource availability and other conditions. 
            </div>
        </div>
      </div> 
      <Footer />
    </div>
  )}
  </NiChartAuthenticator>
  </div>
  );
}

export default Portal;

