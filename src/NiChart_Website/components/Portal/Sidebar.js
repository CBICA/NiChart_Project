import React from 'react';
import styles from '../../styles/Portal_Sidebar.module.css';

function Sidebar({ handleModuleSelection }) {
  return (
    
    <div className={styles.sidebar}>
      <h3 className={styles['sidebar-heading']}>Modules</h3>
      <div className={styles['sidebar-buttons']}>
        <button onClick={() => handleModuleSelection('module1')} className={styles['sidebar-button']}>
          Module 1: Image Processing
        </button>
        <button onClick={() => handleModuleSelection('module2')} className={styles['sidebar-button']}>
          Module 2: Machine Learning
        </button>
        <button onClick={() => handleModuleSelection('module3')} className={styles['sidebar-button']}>
          Module 3: Visualization
        </button>
      </div>
    </div>
  );
}

export default Sidebar;