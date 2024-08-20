import React, { useState } from 'react';
import { useRouter } from 'next/router';
import styles from '../../styles/Components_Sidebar.module.css';

const Sidebar = ({ currentSection, updateExpandedSection }) => {
  const [expandedSection, setExpandedSection] = useState("Reference Dataset");
  const router = useRouter();
  const handleItemClick = (section) => {
    const element = document.getElementById(section);
    if (element) {
      const headerHeight = 180; // Replace with your header's height
      const offsetPosition = element.offsetTop - headerHeight;
      window.scrollTo({
        top: offsetPosition,
        behavior: "smooth"
      });
    }
  };
  const toggleSection = (section) => {
    // Scroll to the top of the page when a new section is toggled
    window.scrollTo({
      top: 0,
      behavior: "smooth"
    });
    setExpandedSection(expandedSection === section ? null : section);
    router.push(`/components#${section}`, undefined, { shallow: true });
    updateExpandedSection(section);
  };


  return (
    <aside className={styles.sidebar}>
      <nav>
        <ul>
          <li className={styles.collapsibleSection}>
            <a onClick={() => toggleSection('Reference Dataset')}>
              <span className={currentSection === 'Reference Dataset' ? styles.rotated : ''}></span>
              <a className={styles.referenceDataset}>Reference Dataset</a>
            </a>
            {currentSection === 'Reference Dataset' && (
              <ul className={styles.innerSection}>
                <li><a onClick={() => handleItemClick('RefVars')}>Demographics and Clinical Variables</a></li>
              </ul>
            )}
          </li>
          <li className={styles.collapsibleSection}>
            <a onClick={() => toggleSection('Image Processing')}>
              <span className={currentSection === 'Image Processing' ? styles.rotated : ''}></span><a className={styles.imageProcessing}>Image Processing</a>
            </a>
            {currentSection === 'Image Processing' && (
              <ul className={styles.innerSection}>
                <li><a onClick={() => handleItemClick('sMRIProcessing')}>sMRI</a></li>
                <li><a onClick={() => handleItemClick('DTIProcessing')}>DTI</a></li>
                <li><a onClick={() => handleItemClick('fMRIProcessing')}>fMRI                
                </a></li>
              </ul>
            )}
          </li>
          <li className={styles.collapsibleSection}>
            <a onClick={() => toggleSection('Harmonization')}>
              <span className={currentSection === 'Harmonization' ? styles.rotated : ''}></span><a className={styles.harmonization}>Data Harmonization</a>
            </a>
            {currentSection === 'Harmonization' && (
              <ul className={styles.innerSection}>
                <li><a onClick={() => handleItemClick('combat_family')}>Combat Family</a></li>
                <li><a onClick={() => handleItemClick('combat_tools')}>Complementary Tools</a></li>
              </ul>
            )}
          </li>
          <li className={styles.collapsibleSection}>
            <a onClick={() => toggleSection('Machine Learning Models')}>
              <span className={currentSection === 'Machine Learning Models' ? styles.rotated : ''}></span>
              <a className={styles.machineLearning}>Machine Learning Models</a>
            </a>
            {currentSection === 'Machine Learning Models' && (
              <ul className={styles.innerSection}>
                <li><a onClick={() => handleItemClick('ml_supervised')}>Supervised Models</a></li>
                <li><a onClick={() => handleItemClick('ml_semisupervised')}>Semi-Supervised Models</a></li>
              </ul>
            )}
          </li>
          <li className={styles.collapsibleSection}>
            <a onClick={() => toggleSection('Data Visualization')}>
              <span className={currentSection === 'Data Visualization' ? styles.rotated : ''}></span>
              <a className={styles.dataVisualization}>Data Visualization</a>
            </a>
            {currentSection === 'Data Visualization' && (
              <ul className={styles.innerSection}>
                <li><a onClick={() => handleItemClick('datavis_viewer')}>NiChart Viewer</a></li>
                <li><a onClick={() => handleItemClick('datavis_webviewer')}>NiChart Web Viewer</a></li>
              </ul>
            )}
          </li>
          <li className={styles.collapsibleSection}>
            <a onClick={() => toggleSection('Deployment')}>
              <span className={currentSection === 'Deployment' ? styles.rotated : ''}></span>
              <a className={styles.deployment}>Deployment</a>
            </a>
            {currentSection === 'Deployment' && (
              <ul className={styles.innerSection}>
                <li><a onClick={() => handleItemClick('deploy_install')}>Software Packages</a></li>
                <li><a onClick={() => handleItemClick('deploy_container')}>Software Containers</a></li>
                <li><a onClick={() => handleItemClick('deploy_cloud')}>Portal</a></li>
              </ul>
            )}
          </li>
        </ul>
      </nav>
    </aside>
  );
};

export default Sidebar;
