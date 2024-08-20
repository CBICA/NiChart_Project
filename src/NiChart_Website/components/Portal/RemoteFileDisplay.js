import { React, useState, useEffect } from 'react';
import { Flex, Heading, Divider, Text, ScrollView, Collection } from '@aws-amplify/ui-react';
import styles from '../../styles/Portal_Module_1.module.css'
import { listBucketContentsForUser, deleteKeyForUser, getKeyMetadata } from '../../utils/uploadFiles.js'
import { ResponsiveButton as Button } from '../Components/ResponsiveButton.js'
// This widget uses the user's Cognito credentials to list that user's bucket contents for a given bucket.
// It also provides functionality to individually delete/remove those contents.

export const RemoteFileDisplay = ({bucket}) =>  {
    
    let [remoteFiles, setRemoteFiles] = useState({});
    let [justCreated, setJustCreated] = useState(true);
    let [numberOfFiles, setNumberOfFiles] = useState(0);
    let [numberOfArchives, setNumberOfArchives] = useState(0);
    let [numberOfScans, setNumberOfScans] = useState(0);
    
    async function update() {
      console.log("Updating RemoteFileDisplay")
      var out_list = await listBucketContentsForUser(bucket)
      if (!out_list) {
        alert ("We couldn't find anything to browse in your bucket. If this is in error, please contact nichart-devs@cbica.upenn.edu.")
      }
      //console.log("out_list", out_list)

      await Promise.all(out_list.map(async (element) => {
        //console.log("RFS Update: element:", element)
        element.meta = await getKeyMetadata(bucket, element.key);
      }))

      //console.log("out_list", out_list)
      var out_dict = {};
      for (const item of out_list) {
        //console.log("item:", item)
        const key = item.key;
        //console.log("Key for this file: ", key)
        out_dict[item.key] = item;
      }
      //console.log("out_dict:", out_dict)
      setRemoteFiles(out_dict)
      
      var n_archives = 0;
      var n_scans = 0;
      var n_files = 0;
      if (out_list === undefined || out_list.length == 0) {
        return;
      }
      for (const item of out_list) {
          n_files += 1;
          if (fileIsArchive(item.key)) {
              n_archives += 1;
          }
          if (fileIsImage(item.key)) {
              n_scans += 1;
          }
        }
      setNumberOfFiles(n_files)
      setNumberOfArchives(n_archives)
      setNumberOfScans(n_scans)
      console.log(remoteFiles)
    } 
    
    async function deleteKeyFromBucket(key) {
        await deleteKeyForUser(bucket, key)
        console.log("Deleting key based on RFD selection")
        //alert("Placeholder From RemoteFileDisplay: User attempting to delete key " + key);
        update()
    }
    
    function fileIsArchive (key) {
        return key.endsWith(".zip") ? true 
           : key.endsWith(".tar.gz") ? true
           : key.endsWith(".tar")? true
           : false
    }
    
    function fileIsImage (key) {
        return key.endsWith(".nii.gz") ? true
        : key.endsWith(".nii") ? true
        : false
    }
    
    function fileIsMacThumbnail (key) {
        return key.toLowerCase().includes("_macosx")
    }

    function fileIsCSV (key) {
        return key.endsWith(".csv")
    }
    
    function getFileStatus (key) {
        if (fileIsMacThumbnail(key)) {
            return "macOS thumbnail file (will not be processed)"
        }
        if (fileIsArchive(key)) {
            //const meta = await getKeyMetadata(bucket, key)
            const meta = remoteFiles[key].meta.metadata
            if (meta['archive_status'] == 'EXTRACTED') {
                return (<font color="green">Archive (Extracted)</font>)
            }
            else if (meta['archive_status'] == 'FAILED') {
                return (<font color="red">Failed to Extract</font>)
            }
            else {
                return "Archive (extraction pending)" 
            }
        }
        else if (fileIsImage(key)) {
            //const meta = await getKeyMetadata(bucket, key)
            const meta = remoteFiles[key].meta.metadata
            if (meta['qc_status'] == 'SUCCEEDED') {
                return (<font color="green">Image (QC Passed)</font>)
            }
            else if (meta['qc_status'] == 'FAILED'){
                return (<font color="red">QC Failed: + {meta['qc_reason']}</font>)
            }
            else {
                return (<font>Image (status pending)</font>)
            }
        }
        else {
            return "N/A";
        }
    }
    
   useEffect(() => {
    const interval =  setInterval(() => {
        update();
        //alert("updating RemoteFileDisplay");
    }, 10000);

    return () => clearInterval(interval);
    }, [remoteFiles]);
    
    if (justCreated) {
        update();
        setJustCreated(false);
    }
    
    return (
        <div>
        <Text><b>Please note</b> that even if your scans fail our quality control checks, you can still attempt to run image processing on them. However, we cannot make any guarantees about the quality of results from data that fails these checks.</Text>
        <Divider orientation="horizontal" />
        <h2>Successfully uploaded scans (scroll to view):</h2>
            <ScrollView height='400px' width='100%'> 
                <Collection 
                    items={Object.entries(remoteFiles)}
                    type="list"
                    direction="column"
                    gap="10px"
                    wrap="nowrap"
                >
                {([key, item], index) => (
                    <div>
                    <Flex direction={{ base: 'row' }} width="100%" justifyContent="space-between" height="10%">
                    <Text>File key: {item.key}</Text>
                    <Text>Type: { fileIsMacThumbnail(item.key)? "macOS Thumbnail (not used)" : fileIsArchive(item.key)? "Archive" : fileIsImage(item.key)? "Scan" : fileIsCSV(item.key)? "Tabular" : "Other"}</Text>
                    <Text>Status: { getFileStatus (item.key) }</Text>
                    <Button loadingText="Deleting..." variation="destructive" onClick={async () => {deleteKeyFromBucket(item.key)}}>Delete</Button>
                    </Flex>
                    <Divider />
                    </div> 
                    )}
                </Collection>
            </ScrollView>
        <p><b>Total files: {numberOfFiles} ({numberOfArchives} archives, {numberOfScans} scans)</b></p>
        <Divider orientation="horizontal" />
        </div>
    )
}
