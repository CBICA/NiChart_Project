# This Python file uses the following encoding: utf-8
"""
contact: software@cbica.upenn.edu
Copyright (c) 2018 University of Pennsylvania. All rights reserved.
Use of this source code is governed by license located in license file: https://github.com/CBICA/NiBAx/blob/main/LICENSE
"""

import argparse
import sys

import pkg_resources  # part of setuptools
from NiChart_DLMUSE import Structural

VERSION = pkg_resources.require("NiChart_DLMUSE")[0].version

def main():
    prog = "NiChart_DLMUSE"
    description = "niCHART Data Preprocessing Pipelines"
    usage = """
    NiChart_DLMUSE v{VERSION}
    ICV calculation, brain segmentation, and ROI extraction pipelines for 
    structural MRI data.

    required arguments:
        [INDIR]         The filepath of the directory containing the input. The 
        [-i, --indir]   input can be a single .nii.gz (or .nii) file or a  
                        directory containing .nii.gz files (or .nii files). 

        [OUTDIR]        The filepath of the directory where the output will be
        [-o, --outdir]  saved.

        [PIPELINETYPE]  Specify type of pipeline[structural, dti, fmri]. 
        [-p,            Currently only structural pipeline is supported.
        --pipelinetype]

        [DERIVED_ROI_MAPPINGS_FILE]     The filepath of the derived MUSE ROI 
        [--derived_ROI_mappings_file]   mappings file.

        [MUSE_ROI_MAPPINGS_FILE]    The filepath of the MUSE ROI mappings file.
        [--MUSE_ROI_mappings_file]
    
    optional arguments: 
        [DLICVMDL]      The filepath of the DLICV model will be. In case the
        [--DLICVmdl]    model to be used is an nnUNet model, the filepath of
                        the model's parent directory should be given. Example: 
                        /path/to/nnUNetTrainedModels/nnUNet/
        
        [DLMUSEMDL]     The filepath of the DLMUSE model will be. In case the
        [--DLMUSEmdl]   model to be used is an nnUNet model, the filepath of
                        the model's parent directory should be given. Example:
                        /path/to/nnUNetTrainedModels/nnUNet/

        [NNUNET_RAW_DATA_BASE]   The filepath of the base directory where the 
        [--nnUNet_raw_data_base] raw data of are saved.  This argument is only 
                                 required if the DLICVMDL and DLMUSEMDL 
                                 arguments are corresponding to a  nnUNet model 
                                 (v1 needs this currently).

        [NNUNET_PREPROCESSED]   The filepath of the directory where the 
        [--nnUNet_preprocessed] intermediate preprocessed data are saved. This
                                argument is only required if the DLICVMDL and
                                DLMUSEMDL arguments are corresponding to a
                                nnUNet model (v1 needs this currently).

        [MODEL_FOLDER]          THIS IS ONLY NEEDED IF BOTH DLICV AND DLMUSE 
        [--model_folder]        MODELS ARE NNUNET MODELS. The filepath of the
                                directory where the models are saved. The path
                                given should be up to (without) the nnUNet/ 
                                directory. Example:
                                /path/to/nnUNetTrainedModels/          correct
                                /path/to/nnUNetTrainedModels/nnUNet/   wrong
                                This is a temporary fix, and will be removed 
                                in the future. Both models should be saved in 
                                the same directory. Example:
                                /path/to/nnUNetTrainedModels/nnUNet/Task_001/
                                /path/to/nnUNetTrainedModels/nnUNet/Task_002/

        [DLICV_TASK]            The task number of the DLICV model. This 
        [--DLICV_task]          argument is only required if the DLICVMDL is a 
                                nnUNet model.

        [DLMUSE_TASK]           The task number of the DLMUSE model. This 
        [--DLMUSE_task]         argument is only required if the DLMUSEMDL is a 
                                nnUNet model.

        [DLICV_FOLD]            The fold number of the DLICV model. This 
        [--DLICV_fold]          argument is only required if the DLICVMDL is a
                                nnUNet model.

        [DLMUSE_FOLD]           The fold number of the DLMUSE model. This
        [--DLMUSE_fold]         argument is only required if the DLMUSEMDL is a
                                nnUNet model.

        [ALL_IN_GPU]            If this var is set, all the processes will be
        [--all_in_gpu]          done in the GPU. This var is only available if 
                                the DLICVMDL and DLMUSEMDL arguments are 
                                corresponding to a nnUNet model. Either 'True',
                                'False' or 'None'. 

        [DISABLE_TTA]           If this var is given, test-time augmentation  
        [--disable_tta]         will be disabled. This var is only available if 
                                the DLICV and DLMUSE models are nnUNet models. 

        [MODE]                  The mode of the pipeline. Either 'normal' or
        [--mode]                'fastest'. 'normal' mode is the default mode.

        [EXTRACT_ROI_MASKS]     Whether the pipeline should extract the individual
        [--extract_roi_masks]   ROI masks or not. By default individual ROI masks
                                are NOT extracted. If passed as an argument, 
                                pipeline will create masks for each individual ROI
                                mask.
    
        [-h, --help]    Show this help message and exit.
        
        [-V, --version] Show program's version number and exit.

        EXAMPLE USAGE:
        
        NiChart_DLMUSE  --indir                     /path/to/input     \
                        --outdir                    /path/to/output    \
                        --pipelinetype structural                      \
                        --derived_ROI_mappings_file /path/to/file.csv  \
                        --MUSE_ROI_mappings_file    /path/to/file.csv  \
                        --nnUNet_raw_data_base      /path/to/folder/   \
                        --nnUNet_preprocessed       /path/to/folder/   \
                        --model_folder              /path/to/folder/   \
                        --all_in_gpu True                              \
                        --mode fastest                                 \
                        --disable_tta
    """.format(VERSION=VERSION)

    parser = argparse.ArgumentParser(prog=prog,
                                     usage=usage,
                                     description=description,
                                     add_help=False)
    
    ################# Required Arguments #################
    # INDIR argument
    parser.add_argument('-i',
                        '--indir', 
                        '--inDir',
                        '--input',
                        type=str, 
                        help='Input T1 image file path.', 
                        default=None, 
                        required=True)
    
    # OUTDIR argument
    parser.add_argument('-o',
                        '--outdir', 
                        '--outDir',
                        '--output',
                        type=str,
                        help='Output file name with extension.', 
                        default=None, required=True)
    
    
    # PIPELINETYPE argument
    parser.add_argument('-p',
                        '--pipelinetype', 
                        type=str, 
                        help='Specify type of pipeline.', 
                        choices=['structural', 'dti', 'fmri'],
                        default='structural', 
                        required=True)
        
    # DERIVED_ROI_MAPPINGS_FILE argument
    parser.add_argument('--derived_ROI_mappings_file', 
                        type=str, 
                        help='derived MUSE ROI mappings file.', 
                        default=None, 
                        required=True)
    
    # MUSE_ROI_MAPPINGS_FILE argument
    parser.add_argument('--MUSE_ROI_mappings_file', 
                        type=str, 
                        help='MUSE ROI mappings file.', 
                        default=None, 
                        required=True)

    ################# Optional Arguments #################
    # DLICVMDL argument
    parser.add_argument('--DLICVmdl', 
                        '--DLICVMDL',
                        type=str, 
                        help='DLICV model path.', 
                        default=None, 
                        required=False)
    
    # DLMUSEMDL argument
    parser.add_argument('--DLMUSEmdl', 
                        '--DLMUSEMDL',
                        type=str, 
                        help='DLMUSE Model path.', 
                        default=None, 
                        required=False)

    # NNUNET_RAW_DATA_BASE argument
    parser.add_argument('--nnUNet_raw_data_base',
                        type=str, 
                        help='nnUNet raw data base.')
    
    # NNUNET_PREPROCESSED argument
    parser.add_argument('--nnUNet_preprocessed',
                        type=str, 
                        help='nnUNet preprocessed.')
    
    # RESULTS_FOLDER argument
    parser.add_argument('--model_folder',
                        type=str,
                        help='Model folder.')
    
    # DLICV_TASK argument
    parser.add_argument('--DLICV_task',
                        type=int, 
                        help='DLICV task.')
    
    # DLMUSE_TASK argument
    parser.add_argument('--DLMUSE_task',
                        type=int, 
                        help='DLMUSE task.')
    
    # DLICV_FOLD argument
    parser.add_argument('--DLICV_fold',
                        type=int, 
                        help='DLICV fold.')
    
    # DLMUSE_FOLD argument
    parser.add_argument('--DLMUSE_fold',
                        type=int, 
                        help='DLMUSE fold.')
    
    # ALL_IN_GPU argument
    parser.add_argument('--all_in_gpu',
                        type=str,
                        default='None',
                        help='All in GPU.')
    
    # DISABLE_TTA argument
    parser.add_argument('--disable_tta',
                        action='store_true',
                        help='Disable TTA.')
    
    # MODE argument
    parser.add_argument('--mode',
                        type=str,
                        default='fastest',
                        choices=['normal', 'fastest'],
                        help='Mode.')
    
    # EXTRACT_ROI_MASKS argument
    parser.add_argument('--extract_roi_masks',
                        action='store_true',
                        help='extract individual ROI masks')
        
    # VERSION argument
    help = "Show the version and exit"
    parser.add_argument("-V", 
                        "--version", 
                        action='version',
                        version=prog+ ": v{VERSION}.".format(VERSION=VERSION),
                        help=help)

    # HELP argument
    help = 'Show this message and exit'
    parser.add_argument('-h', 
                        '--help',
                        action='store_true', 
                        help=help)
    
        
    args = parser.parse_args()

    indir = args.indir
    outdir = args.outdir
    pipelinetype = args.pipelinetype
    derived_ROI_mappings_file = args.derived_ROI_mappings_file
    MUSE_ROI_mappings_file = args.MUSE_ROI_mappings_file
    DLMUSEmdl = args.DLMUSEmdl
    DLICVmdl = args.DLICVmdl
    nnUNet_raw_data_base = args.nnUNet_raw_data_base
    nnUNet_preprocessed = args.nnUNet_preprocessed
    model_folder = args.model_folder
    DLICV_task = args.DLICV_task
    DLMUSE_task = args.DLMUSE_task
    DLICV_fold = args.DLICV_fold
    DLMUSE_fold = args.DLMUSE_fold
    all_in_gpu = args.all_in_gpu
    disable_tta = args.disable_tta
    mode = args.mode
    extract_roi_masks = args.extract_roi_masks


    print()
    print("Arguments:")
    print(args)
    print()



    if(pipelinetype == "structural"):
        Structural.run_structural_pipeline(indir,
                                           DLICVmdl,
                                           DLMUSEmdl,
                                           outdir,
                                           MUSE_ROI_mappings_file,
                                           derived_ROI_mappings_file,
                                           nnUNet_raw_data_base,
                                           nnUNet_preprocessed,
                                           model_folder,
                                           DLICV_task,
                                           DLMUSE_task,
                                           DLICV_fold,
                                           DLMUSE_fold,
                                           all_in_gpu,
                                           disable_tta,
                                           mode,
                                           extract_roi_masks)
        

    elif(pipelinetype == "fmri"):
        print("Coming soon.")
        exit()
    elif(pipelinetype == "dti"):
        print("Coming soon.")
        exit()
    else:
        print("Only [structural, dti and fmri] pipelines are supported.")
        exit()


if __name__ == '__main__':
    main()
