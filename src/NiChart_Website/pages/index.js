import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import Head from 'next/head';
import Header from '../components/Layout/Header';
import Footer from '../components/Layout/Footer';
import Circle from '../components/Components/Circle'
import Favicons from '../components/Favicons/Favicons';
import reportWebVitals from '/utils/reportWebVitals';
import { Typography } from '@mui/material';
import styles from '../styles/index.module.css';

const HomePage = () => {
  const [windowWidth, setWindowWidth] = useState(null);
  const [svgSize, setSvgSize] = useState(225); // Default size
  const [isVertical, setIsVertical] = useState(false); // Layout state

  useEffect(() => {
    const updateWindowWidth = () => {
      setWindowWidth(window.innerWidth);
    };

    updateWindowWidth();

    const handleResize = () => {
      const minSvgSize = 150;
      const maxSvgSize = 250;
      const minWidth = 900;
      const maxWidth = 2160;

      // Linear interpolation for svgSize
      const newSize = Math.min(Math.max(
        minSvgSize + (window.innerWidth - minWidth) * (maxSvgSize - minSvgSize) / (maxWidth - minWidth),
        minSvgSize
      ), maxSvgSize);

      setSvgSize(newSize);
      // Toggle layout based on window width
      setIsVertical(window.innerWidth < 1250);
    };

    window.addEventListener('resize', handleResize);
    handleResize();

    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, []);

  const router = useRouter();

  const handleStepCircleClick = (sectionLabel) => {
    const sectionIdMap = {
      'Curate': 'Reference Dataset',
      'Process': 'Image Processing',
      'Harmonize': 'Harmonization',
      'Learn': 'Machine Learning Models',
      'Visualize': 'Data Visualization',
      'Deploy': 'Deployment'
    };
    const sectionId = sectionIdMap[sectionLabel];
    if (sectionId) {
      router.push(`/components#${encodeURIComponent(sectionId)}`, undefined, { shallow: true });
    }
  };

  const semiCircleAngleSpread = 180; 
  const semiCircleRadius = svgSize * 1.8; 
  const stepCircleDiameter = svgSize;
  const semiCircleHeight = semiCircleRadius * 2 + stepCircleDiameter * 1.5;
  const semiCircleWidth = semiCircleRadius * 2 + stepCircleDiameter * 1.5;

  const StepCircle = ({ label, imageUrl, onClick, strokeColor, index, total, svgSize }) => {
    const angleDeg = (semiCircleAngleSpread / (total - 1)) * index - (semiCircleAngleSpread / 2); 
    const angleRad = (Math.PI / 180) * angleDeg;
    const x = semiCircleRadius * Math.cos(angleRad);
    const y = semiCircleRadius * Math.sin(angleRad); 
    const style = {
      position: 'absolute',
      left: `calc(50% - ${x}px)`,
      top: `calc(50% + ${y}px)`,
      transform: 'translate(0%, -50%)'
    };

    return (
      <div className={styles.stepCircleContainer} style={style}>
        <Circle label={label} imageUrl={imageUrl} onClick={onClick} strokeColor={strokeColor} svgSize={svgSize} />
      </div>
    );
  };

  const steps = [
    { label: "Curate", color: "#a11f25", image: "images/Home/curate.png"},
    { label: "Process", color: "#2152ad", image: "images/Home/process.png"},
    { label: "Harmonize", color: "#92da44", image: "images/Home/harmonize.png"},
    { label: "Learn", color: "#e9a944", image: "images/Home/learn.png"},
    { label: "Visualize", color: "#f5e852", image: "images/Home/visualize.png"},
    { label: "Deploy", color: "#ac29d8", image: "images/Home/deploy.png"},
  ];

  let stepElements;
  if (windowWidth > 1250) {
    stepElements = steps.map((step, index) => (
      <StepCircle
        key={step.label}
        label={step.label}
        imageUrl={step.image}
        onClick={() => handleStepCircleClick(step.label)}
        strokeColor={step.color}
        index={index}
        total={steps.length}
        svgSize={svgSize}
      />
    ));
  } else if (windowWidth > 450) {
    stepElements = steps.map((step) => (
      <Circle
        key={step.label}
        label={step.label}
        imageUrl={step.image}
        onClick={() => handleStepCircleClick(step.label)}
        strokeColor={step.color}
        svgSize={svgSize}
      />
    ));
  }

  return (
    <div className={styles.container}>
      <Head>
        <title>NiChart | Home</title>
        <Favicons />
      </Head>
      <Header />
      <div className={styles.mainContent}>
        <div className={styles.leftSide}>
          <div className={styles.textNiChart}>
            <p><a className={styles.title}>NiChart:</a><a className={styles.text}>Neuro Imaging Chart of AI-based Imaging Biomarkers</a></p>
          </div>
          <div>
          <br></br><br></br><br></br><br></br><br></br>
              <p><a className={styles.smallText}>Want to help shape the future of NiChart? </a></p>
              <br></br>
              <p><a className={styles.smallText}><b>Please take our <a href="https://docs.google.com/forms/d/e/1FAIpQLSddH_eg5RHI94Ph7KYAGibzRSVfXOKReGXbj0Z2YBfF_6c8SA/viewform">5-minute survey</a>!</b></a></p>
            <br></br>
          </div>
          <div className={styles.studentPhoto}>
            <img className={styles.infographic} src="/images/Home/NiChart_info_pic_student_no_background.png" alt="NiChart Infographic"/>
          </div>
        </div>
        <div className={styles.rightSide}>
          {isVertical ? (
            <div className={styles.verticalLayoutContainer}>
              {steps.map((step) => (
                <Circle
                  key={step.label}
                  label={step.label}
                  imageUrl={step.image}
                  onClick={() => handleStepCircleClick(step.label)}
                  strokeColor={step.color}
                  svgSize={svgSize}
                />
              ))}
            </div>
          ) : (
            <div className={styles.semiCircleContainer} style={{ height: `${semiCircleHeight}px` }}>
              {stepElements}
            </div>
          )}
        </div>
      </div>
      <div className={styles.bottom}>
        <div className={styles.bottomDivision}>
          <div>
            <Typography variant='h5'>A framework to process multi-modal MRI images, harmonize to reference data, apply and contribute machine learning models and derive individualized biomarkers called "Neuroimaging Chart Dimensions"</Typography>
          </div>
        </div>
      </div>
      <Footer />
    </div>
  );
};

export default HomePage;

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
