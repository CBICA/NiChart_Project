#import streamlit as st
import os
from math import ceil
import nibabel as nib
import numpy as np
from nibabel.orientations import axcodes2ornt, ornt_transform
import pydicom
from glob import glob
import pandas as pd
import dicom2nifti as dcm
import pydicom

# https://github.com/rordenlab/dcm2niix/blob/master/FILENAMING.md

# https://stackoverflow.com/questions/71042522/conversion-not-working-properly-using-dicom2nifti
# https://pypi.org/project/dicom2nifti/#history
# https://pycad.medium.com/mvp-online-dicom-nifti-viewer-with-python-0da8b3aceadd
# https://github.com/angelomenezes/dicom-labeling-tool

def convert_dicoms_to_nifti(in_dir, out_dir):
    
    # Detect files
    filesandirs = glob.glob(os.path.join(in_dir, '**', '*'), recursive=True)
    files = [f for f in filesandirs if os.path.isfile(f)]
    
    # Read dicom meta data
    dicoms = [pydicom.dcmread(f, stop_before_pixels=True) for f in files]

    #dcm.convert_directory(in_dir, out_dir, compression=True, reorient=True)


#def read_DICOM_slices(path):

    ## Load the DICOM files
    #files = []

    #for fname in glob(path + '*', recursive=False):
        #if fname[-4:] == '.dcm': # Read only dicom files inside folders.
            #files.append(pydicom.dcmread(fname))

    ## Skip files with no SliceLocation
    #slices = []
    #skipcount = 0
    #for f in files:
        #if hasattr(f, 'SliceLocation'):
            #slices.append(f)
        #else:
            #skipcount = skipcount + 1

    #slices = sorted(slices, key=lambda s: s.SliceLocation)

    #img_shape = list(slices[0].pixel_array.shape)
    #img_shape.append(len(slices))
    #img3d = np.zeros(img_shape)

    ## Fill 3D array with the images from the files
    #for i, img2d in enumerate(slices):
        #img3d[:, :, i] = img2d.pixel_array

    #columns = ['PatientID', 'PatientName', 'StudyDescription', 'PatientBirthDate', 'StudyDate', 'Modality', 'Manufacturer', 'InstitutionName', 'ProtocolName']
    #col_dict = {col: [] for col in columns}

    #try:
        #for col in columns:
            #col_dict[col].append(str(getattr(files[0], col)))

        #df = pd.DataFrame(col_dict).T
        #df.columns = ['Patient']
    #except:
        #df = pd.DataFrame([])

    #del files, slices, columns, col_dict

    #return img3d, df

#def display_info(path):
    #columns = ['PatientID', 'PatientName', 'StudyDescription', 'PatientBirthDate', 'StudyDate', 'Modality', 'Manufacturer', 'InstitutionName', 'ProtocolName']
    #col_dict = {col: [] for col in columns}
    #dicom_object = pydicom.dcmread(path + os.listdir(path)[0])

    #for col in columns:
        #col_dict[col].append(str(getattr(dicom_object, col)))

    #df = pd.DataFrame(col_dict).T
    #df.columns = ['Patient']
    #return df
