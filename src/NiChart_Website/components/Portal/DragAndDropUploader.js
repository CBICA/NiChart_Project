import React, { useEffect, useState, useRef} from 'react'
import { Image, ScrollView, Collection, Text, Flex, Divider, Loader, View } from '@aws-amplify/ui-react'
import JSZip from 'jszip'

import { uploadFileMultipart } from '../../utils/uploadFiles'
import { ResponsiveButton as Button } from '../Components/ResponsiveButton'
import { Box } from '@mui/material';
import { CurrencyExchangeOutlined } from '@mui/icons-material'
//import { uploadFile } from '@aws-amplify/ui-react-storage/dist/types/components/StorageManager/utils'

// Turns out Bind doesn't work the way I thought (like in C++) -- we need to curry
function curry(func) {

    return function curried(...args) {
      if (args.length >= func.length) {
        return func.apply(this, args);
      } else {
        return function(...args2) {
          return curried.apply(this, args.concat(args2));
        }
      }
    };
  
  }

const sampleZipProgressCallback = (callback, metadata) => {
    // metadata keys:
    // percent - number - percent of completion (double, 0 - 100)
    // currentFile - string - Name of current file being processed.
    //console.log("sampleZipProgressCallback: currentFile", metadata.currentFile)
    //console.log("sampleZipProgressCallback: percent", metadata.percent)
    /*console.log("zip progress metadata", metadata)
    if (metadata) {
        const finalPercent = metadata.percent.toFixed(2)
        console.log("Sample Zip Callback", metadata.currentFile)
        callback(metadata.currentFile, "Zipping", finalPercent)
    } else {
        return;
    }*/

    
}
const sampleUploadProgressCallback = (callback, filename, progress) => {
    //console.log("Sample Upload Callback", filename)
    // signature of onProgress: ({ transferredBytes, totalBytes })
    const percentage = Math.round(progress.loaded/progress.total) * 100;
    //console.log("sampleUploadProgressCallback, progress (%):", percentage)
    if (percentage == 100) {
        callback(filename, "Complete", percentage);
    } else {
        callback(filename, "Uploading", percentage);
    }
}

const sampleUploadCompleteCallback = (callback, filename, event) => {
    console.log("Upload complate", event)
    callback(filename, "Complete", 100)
}

const sampleUploadErrorCallback = (callback, filename, error) => {
    console.error("Upload error:", error)
    callback(filename, "Failed", 0)
}

const readUploadedFileAsArrayBuffer = (inputFile) => {
    const temporaryFileReader = new FileReader();
  
    return new Promise((resolve, reject) => {
      temporaryFileReader.onerror = () => {
        temporaryFileReader.abort();
        reject(new DOMException("Problem parsing input file."));
      };
  
      temporaryFileReader.onload = () => {
        resolve(temporaryFileReader.result);
      };
      temporaryFileReader.readAsArrayBuffer(inputFile);
    });
  };



export const TestUpload = async (files, updateStatusCallback, updateFileStatusCallback, zipProgressCallback=sampleZipProgressCallback, uploadProgressCallback=sampleUploadProgressCallback, uploadCompleteCallback=sampleUploadCompleteCallback, uploadErrorCallback=sampleUploadErrorCallback) => {
    console.log("Files from drag and drop uploader:", files);

    
    var userWarnedAboutSkip = false;
    var allUploadedArchives = [];
    var looseFileCount = 0;
    

        // Construct zip datestring for bundle filenames
        const d = new Date();
        const year = d.getFullYear();
        const month = d.getMonth() + 1; // WHY
        const date = d.getDate();
        const hour = d.getHours();
        const minute = d.getMinutes();
        const second = d.getSeconds();
        const dateString = month.toString() + "-" + date.toString() + "-" + year.toString();
        const timeString = hour.toString() + "h" + minute.toString() + "m" + second.toString() + "s";
        let miscFilesArchiveFilenameBase = "D" + dateString + "_T" + timeString;

        
    
    // Zipping begins
    updateStatusCallback("Bundling your files, please wait...")
    var fileChunks = [];
    // Chunk files based on size *before* the async loop
    var currentChunk = [];
    var chunkFileSizeTotal = 0;

    for (const file of Array.from(files)) {
        if (file.name.endsWith(".nii.gz") || file.name.endsWith(".nii")) {
            // Add this file to zip object
            console.log("File size", file.size)
            looseFileCount = looseFileCount + 1;
            chunkFileSizeTotal = chunkFileSizeTotal + file.size;
            console.log("chunkFileSizeTotal:", chunkFileSizeTotal)
            const twogigs = 2 * 1000 * 1000 * 1000; // Yes I know it's not accurate, this is for a safety buffer
            console.log("twoGigs: ", twogigs)
            if (chunkFileSizeTotal >= twogigs) {
                console.log("Starting a new chunk")
                fileChunks.push(currentChunk);
                chunkFileSizeTotal = file.size;
                currentChunk = [];
            }
            currentChunk.push(file);
            console.log('Current chunk', currentChunk)
        } else if (file.name.endsWith(".zip")) {
            // Skip this, address it in a separate async loop
            // const buffer = await readUploadedFileAsArrayBuffer(file)
            // // upload this zip by itself
            // allUploadedArchives.push({
            //     name: file.name,
            //     data: buffer,
            // })
        }
        else {
            // alert user that they can't upload this type of file
            console.log("Skipping upload of non-scan, non-archive file clientside:", file)
            if (!userWarnedAboutSkip) {
                alert("Some files you selected were ignored because they were neither zip archives nor NIfTI scans. When your upload finishes, please use the file browser to confirm.");
                userWarnedAboutSkip = true;
            }
        }
    }
    console.log("File chunks:", fileChunks)
    var zipsAlreadyCreated = [];
    var zipsUploadedByUser
    zipsUploadedByUser = await Promise.all(Array.from(files).map(async (file) => {
        if (file.name.endsWith(".zip")) {
           const buffer = await readUploadedFileAsArrayBuffer(file)
            // upload this zip by itself
            //allUploadedArchives.push({
            //    name: file.name,
            //    data: buffer,
            //})
            return {name: file.name, data: buffer}

    } else {
        return "Not a zip"
    }
}))

    zipsAlreadyCreated = zipsUploadedByUser.filter((element) => element instanceof String);


    // Now construct the zip buffers from each chunk
    //var zip = new JSZip();
    var zipsToCreate = [];
    zipsToCreate = await Promise.all(fileChunks.map(async (chunk) => {
        var zip = new JSZip();
        var allBuffers = [];
        allBuffers = await Promise.all(chunk.map(async (file) => {
            const buffer = await readUploadedFileAsArrayBuffer(file)
            return {name: file.name, data: buffer}
        } ))
        allBuffers.forEach((item) => {
            zip.file(item.name, item.data)
        })
        //zipsToCreate.push(zip);
        return zip
    }))

    console.log("zipsToCreate: ", zipsToCreate)
    
    //var zip_count = 0;
    var allNewZips = [];
    allNewZips = await Promise.all(zipsToCreate.map(async (zip, index) => {
        const miscFilesArchiveFilename = miscFilesArchiveFilenameBase + "_" + index.toString() + ".zip"
        const fileBoundZipProgressCallback = curry(zipProgressCallback)(updateFileStatusCallback);
        if (looseFileCount > 0) {
            updateFileStatusCallback(miscFilesArchiveFilename, "Zipping", 0)
        }
        // Now create the zip blobs
        let fullArchiveBlob = await zip.generateAsync({type: "uint8array"}, fileBoundZipProgressCallback);
        return {name: miscFilesArchiveFilename, data: fullArchiveBlob}
        //allUploadedArchives.push({name: miscFilesArchiveFilename, data: fullArchiveBlob});
        //zip_count = zip_count + 1;
    }))

    console.log("All new Zips:", allNewZips)
    var allUploadedArchives = allNewZips.concat(zipsAlreadyCreated);

    console.log("Post-zip all-archive list: ", allUploadedArchives);
    // Zipping has ended
    updateStatusCallback("Uploading your files...")
    await Promise.all(allUploadedArchives.map(async (archive) => {
        await updateFileStatusCallback(archive.name, "Uploading", 0)
     }))



    // And perform the multipart uploads
    var allUploadResults = [];
    /*
    const fileBoundUploadProgressCallback = curry(uploadProgressCallback)(updateFileStatusCallback, miscFilesArchiveFilename);
    const fileBoundUploadCompleteCallback = curry(uploadCompleteCallback)(updateFileStatusCallback, miscFilesArchiveFilename);
    const fileBoundUploadErrorCallback = curry(uploadErrorCallback)(updateFileStatusCallback, miscFilesArchiveFilename);
    if (looseFileCount > 0) {
        await updateFileStatusCallback(miscFilesArchiveFilename, "Uploading", 0);
        let uploadResult = await uploadFileMultipart("cbica-nichart-inputdata", miscFilesArchiveFilename, fullArchiveBlob, fileBoundUploadProgressCallback, fileBoundUploadCompleteCallback, fileBoundUploadErrorCallback);
        allUploadResults.push(uploadResult);
    }*/
    

    

    await Promise.all(allUploadedArchives.map(async (archive) => {
        
        const fileBoundUploadProgressCallback = curry(uploadProgressCallback)(updateFileStatusCallback, archive.name);
        const fileBoundUploadCompleteCallback = curry(uploadCompleteCallback)(updateFileStatusCallback, archive.name);
        const fileBoundUploadErrorCallback = curry(uploadErrorCallback)(updateFileStatusCallback, archive.name);
        let uploadResult = await uploadFileMultipart("cbica-nichart-inputdata", archive.name, archive.data, fileBoundUploadProgressCallback, fileBoundUploadCompleteCallback, fileBoundUploadErrorCallback)
        allUploadResults.push(uploadResult);
    }
    ));
    var allUploadsSuccessful = true;
    for (const result of allUploadResults) {
        // Check that result isn't an error
        if (result instanceof Error) {
            allUploadsSuccessful = false;
        }
    }
    if (allUploadsSuccessful) {
        updateStatusCallback("All uploads finished successfully!")
    } else {
        updateStatusCallback("Some uploads were unsuccessful. Try uploading again.")
    }
    
    
}


export const DragAndDropUploader = ({onUpload=TestUpload}) => {
    const drop = React.useRef(null);
    const [files, setFiles] = useState({
       // "PlaceholderFile1.nii.gz": {name: "PlaceholderFile1.nii.gz", status: "Uploading", percentage: "50" },
       // "PlaceholderFile2.nii.gz": {name: "PlaceholderFile2.nii.gz", status: "Complete", percentage: "100" },
      //  "PlaceholderFile3.nii.gz": {name: "PlaceholderFile3.nii.gz", status: "Failed", percentage: "0" },
});
    const [overallStatus, setOverallStatus] = useState("Waiting for uploads...");
    

    const handleDragOver = (e) => {
        e.preventDefault();
        e.stopPropagation();
    };

    const handleDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();

        const {files} = e.dataTransfer;

        if (files && files.length) {
            onUpload(files, updateOverallStatus, updateFileStatus);
        }
    };

    const updateFileStatus = (filename, status, percentage) => {
        
        var newObj = {...files};
        //console.log("Updating file status:", filename, status, percentage, newObj)
        newObj[filename] = {name: filename, status: status, percentage: percentage}
        //console.log("Newer obj:", newObj)
        setFiles(prevState => {
            return {...prevState, ...newObj};
        });
    }

    const updateOverallStatus = async (status) => {
        setOverallStatus(status)
    }

    useEffect(() => {
        drop.current.addEventListener('dragover', handleDragOver);
        drop.current.addEventListener('drop', handleDrop);
    
        return () => {
            if (drop && drop.current) {
                drop.current.removeEventListener('dragover', handleDragOver);
                drop.current.removeEventListener('drop', handleDrop);
            }

        };
    }, []);

    // useEffect(() => {
    //     const inputElement = document.getElementById("fileElem");
    //     inputElement.addEventListener("change", onUpload, false);
    // }, []);
    const hiddenFileInput = useRef(null);

    const handleClick = event => {
        hiddenFileInput.current.click();
    }
    
    const handleChange = event => {
        const files = event.target.files;
        onUpload(files, updateOverallStatus, updateFileStatus);
    }
    
    return (
        <div
          ref={drop}
        >
        <Flex border="2px #c3c3c3 dashed" justifyContent="center" alignContent="space-around" alignItems="center" padding="3rem" direction="column">
            <View >
                <b>Drag and drop NIfTI files and .zip archives here</b>
            </View>
            <input
            type="file"
            onChange={handleChange}
            ref={hiddenFileInput}
            style={{display: 'none'}} // Make the file input element invisible
            multiple
            />
            <label htmlFor="contained-button-file">
            <Button onClick={handleClick} variant="primary" variation="primary" color="primary" component="span">
                Or browse individual files
            </Button>
            </label>
        </Flex>
        <Box>
        <h3>Upload Tracker</h3>
        <Flex direction="row" justifyContent="space-between" width="100%" height="30%">
            <Text><b>File</b></Text>
            <Text><b>Status</b></Text>
        </Flex>
        <ScrollView maxHeight="250px">
            <Collection
                 items={Object.values(files)}
                 type="list"
            >
                {
                    (item, index) => (
                        <>
                        <Flex direction="row" justifyContent="space-between" width="100%" height="25px">
                            <Text>{item.name}</Text>
                            <Text>{item.status}: {item.percentage}% { item.status == "Complete"? ":)" : item.status == "Uploading" || item.status == "Zipping" ? <Loader /> : "X"}</Text>
                        </Flex>
                        <Divider orientation="horizontal" />
                        </>
                    )
                }

            </Collection>
        </ScrollView>
        </Box>
        </div>
    )
}

