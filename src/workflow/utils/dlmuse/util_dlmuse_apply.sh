#! /bin/bash -x

echo "About to run: $0 $@"

## Read input
in_csv=$1
out_csv=$2

## Prep init data
in_dir=$(dirname ${in_csv})
out_dir=$(dirname ${out_csv})

tmp_dir="${out_dir}/tmprun_dlmuse"
mkdir -pv "${tmp_dir}/nnUNet_preprocessed"
mkdir -pv "${tmp_dir}/nnUNet_raw_database"
mkdir -pv "${tmp_dir}/nnUNet_out"
ln -s `realpath ./nnUNet_model` ${tmp_dir}/nnUNet_model
ln -s `realpath ${in_dir}/Images` "${tmp_dir}/nnUNet_raw_database/nnUNet_raw_data"
droi=`realpath ../../../Image_Processing/sMRI/NiChart_DLMUSE/shared/dicts/MUSE_mapping_derived_rois.csv`
roi=`realpath ../../../Image_Processing/sMRI/NiChart_DLMUSE/shared/dicts/MUSE_mapping_consecutive_indices.csv`

## Apply dlmuse test
cmd="NiChart_DLMUSE --indir ${tmp_dir}/nnUNet_raw_database/nnUNet_raw_data --outdir ${tmp_dir}/nnUNet_out --pipelinetype structural --derived_ROI_mappings_file $droi --MUSE_ROI_mappings_file $roi --nnUNet_raw_data_base ${tmp_dir}/nnUNet_raw_database --nnUNet_preprocessed ${tmp_dir}/nnUNet_preprocessed --model_folder ${tmp_dir}/nnUNet_model --all_in_gpu True --mode fastest --disable_tta"
echo "About to run: $cmd"
$cmd

## Prep final data
mv ${tmp_dir}/nnUNet_out/results_muse_rois ${out_dir}/
for ftmp in $(ls -1 ${out_dir}/results_muse_rois/*_DLMUSE_Volumes.csv); do
    if [ ! -e ${out_csv} ]; then
        cp ${ftmp} ${out_csv}
    else
        tail -1 ${ftmp} >> ${out_csv}
    fi
done
