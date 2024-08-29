#!/bin/sh
# script for execution of deployed applications
#
# Sets up the MATLAB functions (compiled or not) for the current $ARCH and executes 
# the specified command.
#
# Modified by Yuncong Ma, 9/6/2022 

############################################################
help()
{
cat << HELP

# The following script runs a compiled matlab function
#############################################################
Usage : $0 [OPTIONS]
OPTIONS:
Reqd:	
	-m   < dir >    package_matlab   --  path to perform addpath in MATLAB
	-f   < str >    function_matlab  --  file path to the compiled matlab function
	-p	 < str >	parameter   	 --  all input parameters combined in a string
	-log < file >   file_log         --  file path for log

ERROR: Not enough input arguments!!
HELP
exit 1
}

parse()
{
	while [ -n "$1" ];
	do
		case $1 in
			-h)
				help;
				shift 1;;
			-m)
				package_matlab=$2;
				shift 2;;
			-f)
				function_matlab=$2;
				shift 2;;
			-p)
				parameter=$2;
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
	echo -e "Start time                                   : `date +%F-%H:%M:%S`\n"
    echo run_matlab_function.sh
	echo $package_matlab
    echo $function_matlab
    echo $parameter
    echo $file_log
fi

# Organize input parameters
list_package=$(echo $package_matlab | tr "~" "\n")
args0=
for i in $list_package
do
    arg0=${arg0}"addpath(genpath('"${i}"'));"
done
echo $arg0

# Organize input parameters
list_parameter=$(echo $parameter | tr "~" "\n")
args=
for i in $list_parameter
do
    arg=${arg}"'"${i}"',"
done
arg=${arg:0:-1}
echo $arg

# Run matlab
echo -e "\nRun in MATLAB:\n${arg0}${function_matlab}(${arg});exit\n"

matlab -nodisplay -nosplash -r "${arg0}${function_matlab}(${arg});exit"