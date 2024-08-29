#!/bin/sh
# script for execution of deployed applications
#
# Sets up the MATLAB functions (compiled or not) for the current $ARCH and executes 
# the specified command.
#
# Modified by Yuncong Ma, 9/16/2022 

############################################################
help()
{
cat << HELP

# The following script runs a compiled matlab function
#############################################################
Usage : $0 [OPTIONS]
OPTIONS:
Reqd:	
	-rt  < run >    matlab_runtime   --  path for matlab runtime, set to "/cbica/software/external/matlab/R2018A" in default
	-f   < file >   file_MATLAB_bash --  file path to the bash file (run_*.sh) which runs compiled matlab functions
	-p	 < str >	parameter   	 --  all input parameters combined in a string
	-t   < str >    thread           --  number of thread to run MATLAB
	-log < file >   file_log         --  file path for log

ERROR: Not enough input arguments!!
HELP
exit 1
}

parse()
{
	matlab_runtime="/cbica/software/external/matlab/R2018A"
	while [ -n "$1" ];
	do
		case $1 in
			-h)
				help;
				shift 1;;
			-rt)
				matlab_runtime=$2;
				shift 2;;
			-f)
				file_MATLAB_bash=$2;
				shift 2;;
			-p)
				parameter=$2;
				shift 2;;
			-t)
				thread=$2;
				shift 2;;
			-log)
				file_log=$2;
				shift 2;;
			-*)
				echo "ERROR:no such option $1"
				help;;
			*)
				break;;
		esac
	done
	if [ -z "$thread" ]
	then
		thread=1
	fi
}

if [ $# -lt 1 ]
then
	help
fi

## Reading arguments
parse $*

Debug=1
if [ "$Debug" = "1" ]
then
	echo -e "Start time                                   : `date +%F-%H:%M:%S`\n" >> $file_log
    echo -e "\nrun_matlab_compiled.sh\n" >> $file_log
    echo -e "Job: $JOB_ID\n" >> ${file_log}
	echo $matlab_runtime >> $file_log
    echo $file_MATLAB_bash >> $file_log
    echo $parameter >> $file_log
	echo $thread >> $file_log
    echo $file_log >> $file_log
fi

if [ ! -d "${matlab_runtime}" ]
then
    echo -e '\nError: cannot find the matlab runtime at '$matlab_runtime'\n' >> $file_log
	exit 0
fi

# Organize input parameters
list_parameter=$(echo $parameter | tr "~" "\n")
args=
for i in $list_parameter
do
    arg=${arg}" '"${i}"' "
done
echo $arg >> $file_log

cmd=${file_MATLAB_bash}" "${matlab_runtime}" "${arg}" > '"${file_log}"' 2>&1"

echo $cmd >> $file_log

# OMP_THREAD_LIMIT=1 in default
OMP_THREAD_LIMIT=$thread
KMP_ALL_THREADS=$thread

eval ${cmd}

