import ROIDict from '../public/content/Portal/Visualization/Dicts/MUSE_ROI_complete_list.json';

// Extract IDs from the ROI, handling composite ROIs
export function getRelevantROIs(selectedROI) {
    const selectedROIStr = ROIDict[selectedROI]?.Consisting_of_ROIS;
    return selectedROIStr ? selectedROIStr.split(',').map(id => parseInt(id.trim(), 10)) : [];
  }
  
  // Generate colormaps with specified ROIs highlighted
  export function generateColormaps(relevantROIs) {
    const totalROIs = Array.from({ length: 256 }, (_, index) => index);
    let ROI_labels = []
    totalROIs.forEach(individualROI => ROI_labels.push(ROIDict[individualROI]?.Full_Name))
    
    // Initialize colormap arrays
    const colormapTemplate = { R: [], G: [], B: [], A: [], I:totalROIs, labels:ROI_labels, min: 0, max: 255};
    totalROIs.forEach(id => {
      const isIncluded = relevantROIs.includes(id);
      colormapTemplate.R.push(isIncluded ? 255 : 0);
      colormapTemplate.G.push(isIncluded ? 255 : 0);
      colormapTemplate.B.push(isIncluded ? 255 : 0);
      colormapTemplate.A.push(isIncluded ? 255 : 0);
    });
  
    return {
      "custom_red": { ...colormapTemplate, G: colormapTemplate.G.map(() => 0), B: colormapTemplate.B.map(() => 0) },
      "custom_green": { ...colormapTemplate, R: colormapTemplate.R.map(() => 0), B: colormapTemplate.B.map(() => 0) },
      "custom_blue": { ...colormapTemplate, R: colormapTemplate.R.map(() => 0), G: colormapTemplate.G.map(() => 0) },
    };
  }