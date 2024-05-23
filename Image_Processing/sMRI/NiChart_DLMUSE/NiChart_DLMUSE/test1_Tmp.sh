#! /bin/bash

bdir='/home/guraylab/AIBIL/Projects/Tests/NiTest'
t1=${bdir}'/Data/ImgInit/OASIS3/Img/OAS30164_MR_d0233/OAS30164_MR_d0233_T1_LPS.nii.gz'
mrid='OAS30164'
mdlDLICV='/home/guraylab/.deepmrseg/trained_models/dlicv_single/DeepMRSeg_DLICV_v1.0'
mdlMUSE='/home/guraylab/.deepmrseg/trained_models/muse_single/DeepMRSeg_MUSE_v1.0'
rois=${bdir}'/Lists/MUSE_DerivedROIs_Mappings.csv'
outcsv=${bdir}'/Out/test1/OAS30164_muse.csv'

python __main__.py --pipelineType structural --inImg $t1 --DLICVmdl $mdlDLICV --DLMUSEmdl $mdlMUSE --scanID $mrid --derivedROIMappingsFile $rois --outFile $outcsv
