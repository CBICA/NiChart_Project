import os
import re
from pathlib import Path

import nibabel as nib
from nipype.interfaces.base import (BaseInterface, BaseInterfaceInputSpec,
                                    Directory, File, TraitedSpec, traits)

from NiChart_DLMUSE import CombineMasks as combiner
from NiChart_DLMUSE import utils
    
class CombineMasksInputSpec(BaseInterfaceInputSpec):
    in_dir = Directory(mandatory=True, desc='the input dir')
    in_suff = traits.Str(mandatory=False, desc='the input image suffix')
    icv_dir = Directory(mandatory=False, desc='the icv img directory')
    icv_suff = traits.Str(mandatory=False, desc='the icv image suffix')
    out_dir = Directory(mandatory=True, desc='the output dir') 
    out_suff = traits.Str(mandatory=False, desc='the out image suffix')

class CombineMasksOutputSpec(TraitedSpec):
    out_dir = File(desc='the output image')

class CombineMasks(BaseInterface):
    input_spec = CombineMasksInputSpec
    output_spec = CombineMasksOutputSpec

    def _run_interface(self, runtime):

        img_ext_type = '.nii.gz'

        # Set input args
        if not self.inputs.in_suff:
            self.inputs.in_suff = ''
        if not self.inputs.icv_suff:
            self.inputs.icv_suff = ''
        if not self.inputs.out_suff:
            self.inputs.out_suff = '_combined'
        
        ## Create output folder
        if not os.path.exists(self.inputs.out_dir):
            os.makedirs(self.inputs.out_dir)
        
        infiles = Path(self.inputs.in_dir).glob('*' + self.inputs.in_suff + img_ext_type)
        for in_img_name in infiles:
            
            ## Get args
            in_bname = utils.get_basename(in_img_name, self.inputs.in_suff, [img_ext_type])
            icv_img_name = os.path.join(self.inputs.icv_dir, 
                                        in_bname + self.inputs.icv_suff + img_ext_type)
            out_img_name = os.path.join(self.inputs.out_dir,
                                        in_bname + self.inputs.out_suff + img_ext_type)
            
            ## Call the main function
            combiner.apply_combine(in_img_name, icv_img_name, out_img_name)

        # And we are done
        return runtime

    def _list_outputs(self):
        return {'out_dir': self.inputs.out_dir}
