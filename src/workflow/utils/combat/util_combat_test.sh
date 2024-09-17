#! /bin/bash

## Read input
in_csv=$(realpath $1)
in_mdl=$(realpath $2)
out_csv=$3

## Apply combat learn
cmd="neuroharm -a apply -i $in_csv -m $in_mdl -u ${out_csv%.csv}_tmpinit.csv"
echo "About to run: $cmd"
$cmd

## Prepare out file with only harmonized variables
cmd="python src/workflow/utils/combat/util_combat_prep_out.py ${out_csv%.csv}_tmpinit.csv MRID _HARM ${out_csv}"
echo "About to run: $cmd"
$cmd

# rm -rf ${out_csv%.csv}_tmpinit.csv
