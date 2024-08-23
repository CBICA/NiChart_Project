#!/bin/sh

####################################################################################
# Yuncong Ma, 3/11/2023
# Use it as a script or a function to perform the personalized FN computation for FS
# Surface data format, using precomputed gFN
# bash /cbica/home/mayun/Projects/NMF/IPP/Script/Extract_FS_Surface_pFN.sh
# Not finished yet!!
####################################################################################

# For use as a function
parse()
{
    # Default setting
    K=17
    dir_gFN=/cbica/home/mayun/Projects/NMF/Group_FN/FS_Surface
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
file_template=$dir_main/Brain_Template/FS_Surface/Brain_Surface.mat
file_gNb=$dir_main/Brain_Template/FS_Surface/gNb.mat

# Group FN
file_gFN=$dir_main/Group_FN/FS_Surface/gFN_$K.mat

# Data Folder and file list file
dir_data=$dir_main/Data/$exp_name
mkdir -p $dir_data
file_scan=$dir_main/Data/Test2/Preprocessed.mat
file_list=$dir_main/Data/Test2/file_list.txt
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
file_log=$dir_log/Extract_FS_Surface_pFN.log
if test -f "$file_log"
then
    rm ${file_log}
fi

# Start bash processing
echo -e "\nRunning : Extract_FS_Surface_pFN "
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
resId="FS_Surface"

sbjFileSingle=$file_list
file_init=$file_gFN
file_run_matlab_compiled=$dir_main/Code/run_matlab_compiled.sh
MATLAB_function=deployFuncMvnmfL21p1_func_surf_hcp_single_gV_17
MATLAB_bash=$dir_main/Code/run_$MATLAB_function.sh
file_wb=$dir_main/Code/workbench/bin_rh_linux64/wb_command
dir_out=$dir_result
sbjTCFolder=$dir_out

jid=$(qsub -terse -j y -pe threaded 1 -l h_vmem=5G -o $file_log \
    $file_run_matlab_compiled -f $MATLAB_bash -p ${sbjFileSingle}'~'${sbjTCFolder}'~'${file_wb}'~'${file_gNb}'~'${dir_out}'~'${resId}'~'${file_init}'~'${K}'~'${alphaS21}'~'${alphaL}'~'${vxI}'~'${spaR}'~'${ard}'~'${eta}'~'${iterNum}'~'${calcGrp}'~'${parforOn} \
-log $file_log)


# Visualization
dir_figure=$dir_out
file_run_matlab_compiled=${dir_main}/Code/run_matlab_compiled.sh
MATLAB_function=fVisualize_NMF_Step3_HCP_FS_single
MATLAB_bash=${dir_main}/Code/run_${MATLAB_function}.sh

let Flag=0
let N_Flag=1
while [ "$Flag" -lt "$N_Flag" ]
do
    echo "Waiting for Visualization to start at `date +%F-%H:%M:%S`" >> $file_log
    
    file_pFN=$(find $dir_out -type f -name 'final_UV.mat')
    dirarray=($file_pFN)
    let Flag=${#dirarray[@]}
    if [ "$Flag" -lt "$N_Flag" ]
    then
        sleep 60
    fi
done

jid=$(qsub -terse -j y -pe threaded 1 -l h_vmem=5G -o $file_log \
$file_run_matlab_compiled -f $MATLAB_bash -p ${file_template}'~'${file_pFN}'~'${dir_figure} -log ${file_log})

exit
