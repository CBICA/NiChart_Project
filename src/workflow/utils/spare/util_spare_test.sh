#! /bin/bash

## Read input
in_csv=$1
in_mdl=$2
stype=$3
out_csv=$4

## Apply spare test
cmd="spare_score -a test -i $in_csv -m $in_mdl -o ${out_csv%.csv}_tmpout.csv"
echo "About to run: $cmd"
$cmd

## Change column name, select first two columns
sed "s/SPARE_score/SPARE${stype}/g" ${out_csv%.csv}_tmpout.csv | cut -d, -f1,2 > $out_csv
rm -rf ${out_csv%.csv}_tmpout.csv
