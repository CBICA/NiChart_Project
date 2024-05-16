import os
import shutil
from pathlib import Path

from nipype import Node, Workflow
from nipype.interfaces.utility import Merge

# from . import DeepMRSegInterface
from NiChart_DLMUSE import (CalculateROIVolumeInterface,
                              CombineMasksInterface, MaskImageInterface,
                              ReorientImageInterface, ROIRelabelInterface,
                              nnUNetInterface)


def run_structural_pipeline(inDir,
                            DLICVmdl_path,
                            DLMUSEmdl_path,
                            outDir, 
                            dict_MUSE_ROI_Index,
                            dict_MUSE_derived_ROI,
                            nnUNet_raw_data_base=None,
                            nnUNet_preprocessed=None,
                            model_folder=None,
                            DLICV_task=None,
                            DLMUSE_task=None,
                            DLICV_fold=None,
                            DLMUSE_fold=None,
                            all_in_gpu='None',
                            disable_tta=False,
                            mode='fastest',
                            extract_roi_masks=False):
    '''NiPype workflow for structural pipeline
    '''
    
    ##################################
    ## Set init paths and envs
    outDir = os.path.abspath(outDir)
    inDir = os.path.abspath(inDir)
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    
    ## nnUnet-specific settings
    if nnUNet_raw_data_base:
        os.environ['nnUNet_raw_data_base'] = os.path.abspath(nnUNet_raw_data_base) + '/'
    if nnUNet_preprocessed:
        os.environ['nnUNet_preprocessed'] = os.path.abspath(nnUNet_preprocessed) + '/'
    # Assuming that both DLICV and DLMUSE models are in the same folder.
    # Example:
    # 
    # /path/to/nnUNetTrainedModels/nnUNet/Task_001/
    # /path/to/nnUNetTrainedModels/nnUNet/Task_002/
    # 
    # This is not needed if the environment variable is already set.
    if model_folder:
        os.environ['RESULTS_FOLDER'] = os.path.abspath(model_folder) + '/'

    ## Create working dir (FIXME: in output dir for now)
    basedir = os.path.join(outDir,'working_dir')
    if os.path.exists(basedir):
        shutil.rmtree(basedir)
    os.makedirs(basedir, exist_ok=True)

    ##################################
    ## Create nodes
    
    # Create ReorientToLPS Node
    reorientToLPS = Node(ReorientImageInterface.ReorientImage(), name='reorientToLPS')
    reorientToLPS.inputs.in_dir = Path(inDir)
    reorientToLPS.inputs.out_dir = os.path.join(outDir,'out_to_LPS')
    reorientToLPS.inputs.out_suff = '_0000'

    # Create DLICV Node
    # os.environ['RESULTS_FOLDER'] = str(Path(DLICVmdl_path))
    dlicv = Node(nnUNetInterface.nnUNetInference(), name='dlicv')
    dlicv.inputs.out_dir = os.path.join(outDir,'out_dlicv_mask')
    dlicv.inputs.f_val = 1
    if DLICV_fold:
        dlicv.inputs.f_val = DLICV_fold
    dlicv.inputs.t_val = 802
    if DLICV_task:
        dlicv.inputs.t_val = DLICV_task
    dlicv.inputs.m_val = "3d_fullres"
    dlicv.inputs.all_in_gpu = all_in_gpu
    dlicv.inputs.tr_val = "nnUNetTrainerV2"
    dlicv.inputs.mode = mode
    dlicv.inputs.disable_tta = disable_tta

    # Create Apply Mask Node
    maskImage = Node(MaskImageInterface.MaskImage(), name='maskImage')
    maskImage.inputs.in_dir = os.path.join(outDir,'out_to_LPS')
    maskImage.inputs.in_suff = '_0000'
    maskImage.inputs.mask_suff = ''
    maskImage.inputs.out_dir = os.path.join(outDir,'out_dlicv_img')
    maskImage.inputs.out_suff = '_0000'
    
    # Create MUSE Node
    # os.environ['RESULTS_FOLDER'] = str(Path(DLMUSEmdl_path))
    muse = Node(nnUNetInterface.nnUNetInference(), name='muse')
    muse.inputs.out_dir = os.path.join(outDir,'out_muse')
    muse.inputs.f_val = 2
    if DLMUSE_fold:
        muse.inputs.f_val = DLMUSE_fold
    muse.inputs.t_val = 903
    if DLMUSE_task:
        muse.inputs.t_val = DLMUSE_task
    muse.inputs.m_val = "3d_fullres"
    muse.inputs.tr_val = "nnUNetTrainerV2_noMirroring"
    muse.inputs.all_in_gpu = all_in_gpu
    muse.inputs.disable_tta = True # This MUSE model does not support TTA
    muse.inputs.mode = mode

    #create muse relabel Node
    relabel = Node(ROIRelabelInterface.ROIRelabel(), name='relabel')
    relabel.inputs.map_csv_file = os.path.abspath(dict_MUSE_ROI_Index)
    relabel.inputs.in_suff = ''    
    relabel.inputs.out_dir = os.path.join(outDir,'out_muse_relabeled')
    relabel.inputs.out_suff = '_muse_relabeled'    
    
    # Create CombineMasks Node
    combineMasks = Node(CombineMasksInterface.CombineMasks(), name='combineMasks')
    combineMasks.inputs.in_suff = '_muse_relabeled'
    combineMasks.inputs.icv_dir = os.path.join(outDir,'out_dlicv_mask')
    combineMasks.inputs.icv_suff = ''
    combineMasks.inputs.out_dir = os.path.join(outDir,'out_muse_dlicv')
    combineMasks.inputs.out_suff = '_muse_combined'

    # Create ReorientToOrg Node
    reorientToOrg = Node(ReorientImageInterface.ReorientImage(), name='reorientToOrg')
    reorientToOrg.inputs.in_suff = '_muse_combined'
    reorientToOrg.inputs.ref_dir = Path(inDir)
    reorientToOrg.inputs.ref_suff = ''
    reorientToOrg.inputs.out_dir = os.path.join(outDir,'out_muse_orient_orig')
    reorientToOrg.inputs.out_suff = '_muse_orig_orient'

    # Create roi csv creation Node
    roi_to_csv = Node(CalculateROIVolumeInterface.CalculateROIVolume(), name='roi_to_csv')
    roi_to_csv.inputs.in_suff = '_muse_orig_orient'
    roi_to_csv.inputs.list_single_roi = os.path.abspath(dict_MUSE_ROI_Index)
    roi_to_csv.inputs.map_derived_roi = os.path.abspath(dict_MUSE_derived_ROI)
    roi_to_csv.inputs.out_dir = os.path.join(outDir, 'results_muse_rois')
    roi_to_csv.inputs.extract_roi_masks = extract_roi_masks      ## If True, we create an individual mask for each ROI
    roi_to_csv.inputs.out_dir_roi_masks = os.path.join(outDir, 'out_muse_individual_roi_masks')
    

    ##################################
    ## Define workflow

    wf = Workflow(name="structural", base_dir=basedir)
    wf.connect(reorientToLPS, "out_dir", dlicv, "in_dir")
    wf.connect(dlicv, "out_dir", maskImage, "mask_dir")
    wf.connect(maskImage, "out_dir", muse, "in_dir")
    wf.connect(muse, "out_dir", relabel, "in_dir")
    wf.connect(relabel, "out_dir", combineMasks, "in_dir")
    wf.connect(combineMasks, "out_dir", reorientToOrg, "in_dir")
    wf.connect(reorientToOrg,"out_dir", roi_to_csv, "in_dir")
    
    wf.base_dir = basedir
    wf.run()
    print("Exiting function")
