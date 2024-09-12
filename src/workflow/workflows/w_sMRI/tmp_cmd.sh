

p0='Study1'
p1='/home/gurayerus/GitHub/CBICA/NiChart_Project/test/test_input/test2_rois/Study1/Study1_DLMUSE.csv'
p2='/home/gurayerus/GitHub/CBICA/NiChart_Project/test/test_input/test2_rois/Study1/Study1_Demog.csv'
p3='/home/gurayerus/GitHub/CBICA/NiChart_Project/test/test_output/test2_rois'

cmd="python call_snakefile.py --dset_name $p0 --input_rois $p1 --input_demog $p2 --dir_output $p3"
echo $cmd
read -p ee
$cmd
