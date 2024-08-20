import React from 'react';
import styles from '../../styles/Footer.module.css';
import { VERSION, LASTUPDATED } from '../../utils/Version.js';
import { PinDropSharp } from '@mui/icons-material';


const Footer = () => {
  return (
    <footer className={styles.footer}>
      <div className={styles.footerContent}>
        <p>
          © 2024{' '}
          <a href="https://www.med.upenn.edu/cbica/" target="_blank" rel="noopener noreferrer" className={styles.cbicaLink}>
            Center for Biomedical Image Computing and Analytics (CBICA), University of Pennsylvania
          </a>
          . All rights reserved.
          <br></br> Version {VERSION}. Last updated {LASTUPDATED}. <br/>
          <a rel="noopener noreferrer" href="https://www.upenn.edu/about/privacy-policy" className={styles.cbicaLink}> Privacy Policy </a>
             | 
            <a target="_blank" rel="noopener noreferrer" href="https://www.upenn.edu/about/disclaimer" className={styles.cbicaLink}> UPenn Disclaimer </a>
        </p>
      </div>
    </footer>
  );
};

export default Footer;
