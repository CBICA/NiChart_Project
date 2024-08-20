#!/bin/bash

while read requirement;
do 
	if conda install --yes $requirement; then
		echo "Successfully install: ${requirement} from main channel"
	elif conda install --yes -c conda-forge $requirement; then
	  echo "Successfully install: ${requirement} from conda-forge channel"
	else
	  conda install -c ${requirement} pytorch-cpu
	fi
done < requirements.txt
