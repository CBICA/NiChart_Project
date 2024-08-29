#!/bin/sh

####################################################################################
# Yuncong Ma, 3/11/2023
# Use it as a script or a function to perform the personalized FN computation for MNI_Volume
# Surface data format, using precomputed gFN
# bash /cbica/home/mayun/Projects/NMF/IPP/Script/Extract_MNI_Volume_pFN.sh
####################################################################################

# For use as a function
parse()
{
    # Default setting
    K=17
    dir_gFN=/cbica/home/mayun/Projects/NMF/Group_FN/MNI_Volume
    exp_name=Test2
    
    while [ -n "$1" ];
    do
        case $1 in
            -k)
                K=$2;
            shift 2;;
            -exp)
                exp_name=$2;
            shift 2;;
            -*)
                echo "ERROR:no such option $1"
            exit;;
            *)
            break;;
        esac
    done
}

parse $*

# Current dir and main dir
dir_bash=$(dirname -- "$0")
dir_main="$(dirname "$dir_bash")"

# Brain_Template
file_mask=$dir_main/Brain_Template/MNI_Volume/Brain_Mask.nii
file_overlay=$dir_main/Brain_Template/MNI_Volume/Overlay_Image.mat
file_gNb=$dir_main/Brain_Template/MNI_Volume/gNb.mat

# Group FN
file_gFN=$dir_main/Group_FN/MNI_Volume/gFN_$K.mat

# Data Folder and file list file
dir_data=$dir_main/Data/$exp_name
file_scan=$dir_data/filtered_func_data_clean.nii.gz
file_list=$dir_data/file_list.txt
if test -f "$file_list"
then
    rm ${file_list}
fi
echo -e "$file_scan" >> $file_list


# Result Folder
dir_result=$dir_main/Result/$exp_name
mkdir -p $dir_result

# Log File
dir_log=$dir_main/Log
file_log=$dir_log/Extract_MNI_Volume_pFN.log
if test -f "$file_log"
then
    rm ${file_log}
fi

# Start bash processing
echo -e "\nRunning : Extract_MNI_Volume_pFN "
echo -e "Start time : `date +%F-%H:%M:%S`\n"

# Compute pFN
# Set parameter
spaR=1
vxI=0
ard=0 # 1 for strong sparsity
eta=0 # 1 for strong sparsity
iterNum=30
alphaS21=2
alphaL=10
calcGrp=0
parforOn=0
resId="MNI_Volume"

sbjFileSingle=$file_list
file_init=$file_gFN
file_run_matlab_compiled=$dir_main/Code/run_matlab_compiled.sh
MATLAB_function=deployFuncMvnmfL21p1_func_vol_single
MATLAB_bash=$dir_main/Code/run_$MATLAB_function.sh
dir_out=$dir_result
sbjTCFolder=$dir_out

jid=$(qsub -terse -j y -pe threaded 1 -l h_vmem=5G -o $file_log \
    $file_run_matlab_compiled -f $MATLAB_bash -p ${sbjFileSingle}'~'${sbjTCFolder}'~'${file_mask}'~'${file_gNb}'~'${dir_out}'~'${resId}'~'${file_init}'~'${K}'~'${alphaS21}'~'${alphaL}'~'${vxI}'~'${spaR}'~'${ard}'~'${eta}'~'${iterNum}'~'${calcGrp}'~'${parforOn} \
-log $file_log)

