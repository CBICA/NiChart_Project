import React, { useEffect, useState } from 'react';
import dynamic from 'next/dynamic';
import NiiVue from '../../utils/niiViewer.js';
import Modal from 'react-modal';
import { FormControl, InputLabel, Select, MenuItem, Button } from '@mui/material';
import Autocomplete from '@mui/material/Autocomplete';
import TextField from '@mui/material/TextField';
import { breakStringIntoParts } from '../../utils/chartTitleBreaker.js';
import styles from '../../styles/Chart.module.css';
import MUSEROICompleteList from '/public/content/Portal/Visualization/Dicts/MUSE_ROI_complete_list.json';

const PlotlyChart = dynamic(() => import('react-plotly.js'), { ssr: false });

const Chart = ({ name, data, reference, roi, referenceOption, onDelete, onROIChange, onReferenceDataChange }) => {
  const [chartData, setChartData] = useState([]);
  const [chartLayout, setChartLayout] = useState(null);
  const [chartConfig, setChartConfig] = useState(null);
  const [eventHandlers, setEventHandlers] = useState(null);
  const [modalIsOpen, setModalIsOpen] = useState(false); // State for modal visibility
  const [selectedDataPoint, setSelectedDataPoint] = useState(null); // State for clicked data point info
  const [clickedDataPointId, setClickedDataPointId] = useState(null);

  const openModal = (dataPoint) => {
    setSelectedDataPoint(dataPoint);
    setModalIsOpen(true);
  };
  const closeModal = () => {
    setSelectedDataPoint(null);
    setModalIsOpen(false);
  };

  useEffect(() => {
    // Calculate the marker size based on the number of data points
    const calculateMarkerSize = (dataLength) => {
      const minSize = 8;
      const maxSize = 4;
      const minDataLength = 10;
      const maxDataLength = 100;

      // Clamp dataLength between minDataLength and maxDataLength
      const clampedDataLength = Math.min(Math.max(dataLength, minDataLength), maxDataLength);

      // Calculate the interpolation factor (0 to 1) based on dataLength
      const interpolationFactor = (clampedDataLength - minDataLength) / (maxDataLength - minDataLength);

      // Linearly interpolate the marker size between minSize and maxSize
      const markerSize = minSize + interpolationFactor * (maxSize - minSize);
      return Math.floor(markerSize);
    };

    // Convert ROI index (ID) to Full Name
    const roiFullName = MUSEROICompleteList[roi]?.Full_Name || roi;
    const roi_MUSE_ROI_Name =  MUSEROICompleteList[roi]?.MUSE_ROI_Name || roi;
    const filteredData = reference.filter((row) => row.ROI_Name === roi_MUSE_ROI_Name);
    const updatedData = [];
    const referenceDataKeys = ["centile_5", "centile_10", "centile_25", "centile_50", "centile_75", "centile_90", "centile_95"];
    const redColors = referenceDataKeys.map((key, index) => {
      const centerIndex = (referenceDataKeys.length - 1) / 2;
      const distanceFromCenter = Math.abs(index - centerIndex);
      const lightness = 80 + distanceFromCenter * distanceFromCenter * 2;
      return `hsl(0, 100%, ${lightness}%)`;
    });

    // Add reference scatter traces for each centile key
    for (let i = 0; i < referenceDataKeys.length; i++) {
      const key = referenceDataKeys[i];
      updatedData.push({
        // x: filteredData.map((row) => row.Age_At_Visit),
        x: filteredData.map((row) => row.Age),
        y: filteredData.map((row) => row[key]),
        mode: 'line',
        line: {
          color: redColors[i],
        },
        fill: key == "centile_5" ? 'none' : 'tonexty', // Fill area below the line
        fillcolor: redColors[i], // Use the same color for the filled area
        type: 'scatter',
        name: key,
        hovertemplate: '<i>Volume</i>: %{y:.2f}' + '<br><b>Age</b>: %{x:.1f}<br>',
      });
    }
    if (data.length > 0) {
      // Detect the correct column name for IDs and Ages dynamically
      let idColumnName = "ID"; // Default ID column name
      let ageColumnName = "Age"; // Default Age column name
      const columnNames = Object.keys(data[0]); // Get the column names from the first row

      // Check for variations of column names
      const idVariations = ["ID", "id", "MRID", "mrid", "scanid"];
      const ageVariations = ["Age", "age", "age_at_scan", "Age_at_scan"];
      idVariations.forEach(variation => {
        if (columnNames.includes(variation)) {
          idColumnName = variation;
        }
      });
      ageVariations.forEach(variation => {
        if (columnNames.includes(variation)) {
          ageColumnName = variation;
        }
      });


      // Add user data scatter trace
      updatedData.push({
        x: data.map((row) => parseFloat(row[ageColumnName])),
        y: data.map((row) => parseFloat(row[roi])),
        id: data.map((row) => row[idColumnName]),
        type: 'scatter',
        mode: 'markers',
        marker: {
          color: 'blue',
          size: calculateMarkerSize(data.length),
        },
        name: "User data",
        text: data.map((row) => row[idColumnName]),
        hovertemplate: '<b>%{text}</b><br>' + '<b>Volume</b>: %{y:.2f}<br>' + '<b>Age</b>: %{x:.1f}',
      });
    }

    const yValues = data.map((row) => parseFloat(row[roi]));
    const minDataPoint = Math.min(...yValues);
    const maxDataPoint = Math.max(...yValues);
    // Calculate the range based on 25% of the highest and lowest data points
    const dataRange = [minDataPoint * 0.75, maxDataPoint * 1.2];
    // Break the title into smaller parts using the imported function
    const brokenTitle = breakStringIntoParts(name, 65); // Break every 65 seems to work fine
    const brokenROI = breakStringIntoParts(roiFullName, 35); // Break every 35 seems to work fine
    const layout = {
      title: brokenTitle,
      xaxis: {
        title: "Age",
      },
      yaxis: {
        title: brokenROI, // Display the Full Name
        range: dataRange,
      },
      hovermode: 'closest',
      hoverinfo: 'text'
    };

    const config = {
      scrollZoom: true,
      editable: true,
      toImageButtonOptions: {
        format: 'png', // one of png, svg, jpeg, webp
        filename: name,
        height: 1000,
        width: 1000,
        scale: 10 // Multiply title/legend/axis/canvas sizes by this factor
      },
      displaylogo: false,
    };

    const onClickHandler = (event) => {
      if (event.points) {
        const clickedDataPoint = event.points[0];
        const { x, y, curveNumber, pointNumber, data } = clickedDataPoint;
        if ("id" in data) {
          // If ID exists, we have user data.
          const id = data["id"][pointNumber];
          setClickedDataPointId(id);
          // Disable Modal for now. TODO: fix display of scan.
          openModal({
            x,
            y,
            curveNumber,
            id,
          });
        }
      }
    };

    setChartData(updatedData);
    setChartLayout(layout);
    setChartConfig(config);
    setEventHandlers({ click: onClickHandler });
  }, [data.x, data.y, name, reference, roi]);

  return (
    <>
      <div className={styles.visualizationContainer}>
        <PlotlyChart data={chartData}
          layout={chartLayout}
          config={chartConfig}
          onClick={eventHandlers?.click}
        />
        <div className={styles.controlsContainer}>
          <div className={styles.chartControls}>
            <div className={styles.chartControl}>
              <Autocomplete
                value={MUSEROICompleteList[roi]}
                onChange={(event, newValue) => {
                  onROIChange(newValue ? newValue.ID : "");
                }}
                options={Object.values(MUSEROICompleteList)}
                getOptionLabel={(option) => {
                  if (option.ID === "SPARE_score_AD") {
                    return "SPARE AD score";
                  } else if (option.ID === "SPARE_score_BA") {
                    return "SPARE BA score";
                  } else {
                    return `${option.ID}: ${option.Full_Name}`;
                  }
                }}
                renderInput={(params) => (
                  <TextField 
                    {...params} 
                    label="Change ROI" 
                    variant="standard" 
                    style={{ minWidth: '250px' }} // Set a minimum width or adapt as needed
                  />
                )}
              />
            </div>
            <div className={styles.chartControl}>
              {/* Dropdown to select reference data */}
              <FormControl variant="standard" style={{ minWidth: '250px' }}>
                <InputLabel>Change Reference Data</InputLabel>
                <Select value={referenceOption} onChange={(e) => onReferenceDataChange(e.target.value)}>
                    <MenuItem value="CN">CN</MenuItem>
                    <MenuItem value="CN - Female only">CN - Female only</MenuItem>
                    <MenuItem value="CN - Male only">CN - Male only</MenuItem>
                    <MenuItem value="AD">AD</MenuItem>
                    <MenuItem value="AD - Female only">AD - Female only</MenuItem>
                    <MenuItem value="AD - Male only">AD - Male only</MenuItem>
                    <MenuItem value="MCI">MCI</MenuItem>
                    <MenuItem value="MCI - Female only">MCI - Female only</MenuItem>
                    <MenuItem value="MCI - Male only">MCI - Male only</MenuItem>
                </Select>
              </FormControl>
            </div>
          </div>
          {/* Button to destroy the plot */}
          <Button  variant="contained" color="primary" onClick={onDelete}>Close Chart</Button>
        </div>
        {/* Modal to show the NiiVue component */}
      </div>
      <Modal id='niivue_modal'
        isOpen={modalIsOpen}
        onRequestClose={closeModal}
        contentLabel="NiiVue Modal"
        style={{
          content: {
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -40%)', // Center the modal
            width: '80vh', // Responsive width
            height: '75vh', // Responsive height based on viewport height
            overflow: 'hidden visible', // Adds scroll to the modal content if it exceeds the height
            padding: '1%', // Optional: Adds some padding inside the modal
            border: '2px solid #ccc', // Optional: Adds a border to the modal
            borderRadius: '10px', // Optional: Rounds the corners of the modal
            backgroundColor: '#fff', // Optional: Sets a background color for the modal content
          },
          overlay: {
            backgroundColor: 'rgba(0,0,0,0.6)' // Optional: Adds a semi-transparent overlay behind the modal
          }
        }}
        ariaHideApp={false}
      >
        {/* Nii Vue visualization here */}
        {selectedDataPoint && (
          <NiiVue subjectID={clickedDataPointId} roi={roi} closeModal={closeModal}> </NiiVue>
        )}
      </Modal>
    </>
  );
};

export default Chart;
