import React from 'react';
import styles from '../../styles/index.module.css';

const StepCircle = ({ label, imageUrl, onClick, strokeColor, svgSize }) => {
  const outerCircleRadius = svgSize * 0.4167; // Proportional to svgSize
  const innerCircleRadius = svgSize * 0.3333; // Proportional to svgSize
  const strokeWidth = svgSize * 0.0833; // Proportional to svgSize
  const imageSize = innerCircleRadius * 2; // same as innerCircle diameter
  const imagePosition = (svgSize - imageSize) / 2; // to center the image
  const textPathD = `M${svgSize * 0.05},${svgSize / 2} A${outerCircleRadius},${outerCircleRadius} 0 0,0 ${svgSize * 0.95},${svgSize / 2}`;
  const adjustedFontSize = svgSize * (11 / 120); // fontSize as a percentage of svgSize
  const adjustedStrokeWidth = svgSize * (1 / 120); // stroke width as a percentage of svgSize
  
  return (
    <div className={styles.stepCircleContainer} onClick={() => onClick(label)} role="button" tabIndex={0}>
      <svg xmlns="http://www.w3.org/2000/svg" 
              xmlnsXlink="http://www.w3.org/1999/xlink" 
              className={styles.stepCircleSVG} width={svgSize} height={svgSize}>
            {/* Outer Circle */}
            <circle cx={svgSize / 2} cy={svgSize / 2} r={outerCircleRadius} style={{ stroke: strokeColor, strokeWidth }}/>
            {/* Inner Circle */}
            <circle cx={svgSize / 2} cy={svgSize / 2} r={innerCircleRadius} />
            {/* Image Clipping Path */}
            <clipPath id="circle-view">
              <circle cx={svgSize / 2} cy={svgSize / 2} r={innerCircleRadius} />
            </clipPath>
            {/* Image */}
            <image xlinkHref={imageUrl} x={imagePosition} y={imagePosition} height={imageSize} width={imageSize} clipPath="url(#circle-view)" />
            {/* Path for the Curved Text */}
            <path id="text-path" d={textPathD} fill="none" />
            {/* Curved Text */}
            <text className={styles.circleLabel} style={{ fontSize: `${adjustedFontSize}px` , strokeWidth: `${adjustedStrokeWidth}px` }}>
              <textPath xlinkHref="#text-path" startOffset="50%" textAnchor="middle">{label}</textPath>
            </text>
          </svg>
    </div>
  );
};

export default StepCircle;