import { useRef, useEffect, useState } from "react";
import { Niivue, NVImage } from "@niivue/niivue";
import { Button, Select, MenuItem } from "@mui/material";
import { getRelevantROIs, generateColormaps } from '/utils/roiColormaps';
import { downloadOutputFile, downloadInputFile } from "./uploadFiles";
import ROIDict from '../public/content/Portal/Visualization/Dicts/MUSE_ROI_complete_list.json';

const NiiVue = ({ subjectID, roi, closeModal }) => {
  const input_String = `${subjectID}.nii.gz`;
  const DLMUSE_String = `${subjectID}_DLMUSE.nii.gz`;
  const canvas = useRef(null);
  const [isError, setIsError] = useState(false);
  const [overlayColor, setOverlayColor] = useState("custom_blue");
  const [errorMessage, setErrorMessage] = useState("");
  let nv;

  const showHeader = () => {
    if (nv) {
      alert(nv.volumes[0].hdr.toFormattedString());
    }
  };

  const handleLocationChange = (data) => {
    if (nv) {
      const { mm, vox, values } = data;
      const overlayVolume = nv.volumes[1];
      let roiName = "";

      let strHtml = `
        <div>Cursor Location:</div>
        <div>&nbsp;&nbsp;- In millimeters: (${mm[0].toFixed(2)} mm, ${mm[1].toFixed(2)} mm, ${mm[2].toFixed(2)} mm)</div>
        <div>&nbsp;&nbsp;- In voxels: (${vox[0]}, ${vox[1]}, ${vox[2]})</div>
        <div>Values:</div>
        <div>&nbsp;&nbsp;- Raw image: ${values[0].value.toFixed(2)}</div>`;

      if (overlayVolume) {
        const overlayValue = values[1].value.toFixed(0);
        roiName = ROIDict[Math.floor(overlayValue)]?.Full_Name || "";
        if (roiName) {
          roiName = " - <b>" + roiName + "</b>";
        }
        strHtml += `<div>&nbsp;&nbsp;- Overlay: ${overlayValue}${roiName}</div>`;
      }

      document.getElementById("location").innerHTML = strHtml;
    }
  };

  const initializePosition = (imageOptions, volume) => {
    const overlayVolume = nv?.volumes[1];

    if (overlayVolume) {
      const { img: data, dimsRAS: dims } = volume;
      const voxelCount = dims[1] * dims[2] * dims[3];
      let sumX = 0, sumY = 0, sumZ = 0, count = 0;
      const relevantROIs = getRelevantROIs(roi);

      for (let i = 0; i < voxelCount; i++) {
        if (relevantROIs.includes(data[i])) {
          const x = i / (dims[2] * dims[3]);
          const y = (i % (dims[1] * dims[2])) / dims[1];
          const z = (i % (dims[2] * dims[3])) / dims[2];

          sumX += x;
          sumY += y;
          sumZ += z;
          count++;
        }
      }

      if (count > 0) {
        nv.scene.crosshairPos = [
          Math.floor(sumX / count) / dims[1],
          Math.floor(sumY / count) / dims[2],
          Math.floor(sumZ / count) / dims[3]
        ];
      }
    }
  };

  const loadNiftiFiles = async () => {
    try {
      const inputT1 = await downloadInputFile(input_String, false);
      const outputDLMUSE = await downloadOutputFile(DLMUSE_String, false);
      const relevantROIs = getRelevantROIs(roi);
      const colormaps = generateColormaps(relevantROIs);
      const config = {
        crosshairColor: [1, 1, 1, 1],
        show3Dcrosshair: true,
        onLocationChange: handleLocationChange,
        onVolumeAddedFromUrl: initializePosition,
      };

      if (canvas.current) {
        nv = new Niivue(config);
        nv.attachToCanvas(canvas.current);
        Object.keys(colormaps).forEach((key) => {
          nv.addColormap(key, colormaps[key]);
        });

        const inputVolume = await NVImage.loadFromFile({
          file: new File([inputT1], input_String),
          colormap: "gray",
          opacity: 1,
        });

        nv.addVolume(inputVolume);

        try {
          const overlayVolume = await NVImage.loadFromFile({
            file: new File([outputDLMUSE], DLMUSE_String),
            colormap: overlayColor,
            opacity: 0.7,
          });
          nv.addVolume(overlayVolume);
        } catch (error) {
          console.warn("Overlay volume not found:", error.message);
        }
      }
    } catch (error) {
      console.error(error.message);
      setIsError(true);
      setErrorMessage("The original scan file does not exist.");
    }
  };

  const toggleOverlayVisibility = async () => {
    try {
      if (nv) {
        const overlayVolume = nv.volumes[1];

        if (overlayVolume) {
          nv.removeVolumeByIndex(1);
          nv.drawScene();
        } else {
          const outputDLMUSE = await downloadOutputFile(DLMUSE_String, false);
          const newOverlayVolume = await NVImage.loadFromFile({
            file: new File([outputDLMUSE], DLMUSE_String),
            colormap: overlayColor,
            opacity: 0.7,
          });
          nv.addVolume(newOverlayVolume);
          nv.drawScene();
        }
      }
    } catch (error) {
      console.error(error.message);
      setIsError(true);
      setErrorMessage("The overlay scan file does not exist.");
    }
  };

  const handleOverlayColorChange = async (color) => {
    if (nv) {
      const overlayVolume = nv.volumes[1];
      if (overlayVolume) {
        nv.removeVolumeByIndex(1);
        try {
          const outputDLMUSE = await downloadOutputFile(DLMUSE_String, false);
          const newOverlayVolume = await NVImage.loadFromFile({
            file: new File([outputDLMUSE], DLMUSE_String),
            colormap: color,
            opacity: 0.7,
          });
          nv.addVolume(newOverlayVolume);
          nv.drawScene();
          setOverlayColor(color);
        } catch (error) {
          console.error(error.message);
          setIsError(true);
          setErrorMessage("The overlay scan file does not exist.");
        }
      }
    }
  };

  useEffect(() => {
    loadNiftiFiles();
  }, [overlayColor]);

  return (
    <>
      {isError ? (
        <p>Error: {errorMessage}</p>
      ) : (
        <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
          <div style={{ height: '5%', display: 'flex', flexDirection: 'row', gap: '2%' }} id="niivue_controls">
            <Button onClick={toggleOverlayVisibility}>Toggle Overlay</Button>
            <Select
              value={overlayColor}
              onChange={(e) => handleOverlayColorChange(e.target.value)}
            >
              <MenuItem value="custom_blue">Blue - ROI {roi}</MenuItem>
              <MenuItem value="custom_green">Green - ROI {roi}</MenuItem>
              <MenuItem value="custom_red">Red - ROI {roi}</MenuItem>
              <MenuItem value="nih">NIH - all ROIs</MenuItem>
            </Select>
            <Button onClick={() => showHeader()}>Show Header</Button>
          </div>
          <div style={{ height: '70%', width: '80%', alignItems: 'center', margin: '1% 10%' }} id="niivue">
            <canvas ref={canvas} />
          </div>
          <div style={{ height: '15%', padding: '2%' }} id="location">
            <div>Cursor Location:</div>
            <div>&nbsp;&nbsp;- In millimeters: ( -mm, -mm, -mm)</div>
            <div>&nbsp;&nbsp;- In voxels: (-, -, -)</div>
            <div>Values:</div>
            <div>&nbsp;&nbsp;- Raw image: -</div>
          </div>
        </div>
      )}
    </>
  );
};

export default NiiVue;
