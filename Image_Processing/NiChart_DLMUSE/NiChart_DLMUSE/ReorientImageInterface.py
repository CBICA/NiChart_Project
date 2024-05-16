import os
import re
from pathlib import Path

import nibabel as nib
from nipype.interfaces.base import (BaseInterface, BaseInterfaceInputSpec,
                                    Directory, File, TraitedSpec, traits)

from NiChart_DLMUSE import ReorientImage as reorienter
from NiChart_DLMUSE import utils
    
class ReorientImageInputSpec(BaseInterfaceInputSpec):
    in_dir = Directory(mandatory=True, desc='the input dir')
    in_suff = traits.Str(mandatory=False, desc='the input image suffix')
    ref_dir = Directory(mandatory=False, desc='the ref img directory')
    ref_suff = traits.Str(mandatory=False, desc='the ref image suffix')
    out_dir = Directory(mandatory=True, desc='the output dir') 
    out_suff = traits.Str(mandatory=False, desc='the out image suffix')

class ReorientImageOutputSpec(TraitedSpec):
    out_dir = File(desc='the output image')

class ReorientImage(BaseInterface):
    input_spec = ReorientImageInputSpec
    output_spec = ReorientImageOutputSpec

    def _run_interface(self, runtime):

        img_ext_type = '.nii.gz'

        # Set input args
        if not self.inputs.in_suff:
            self.inputs.in_suff = ''
        if not self.inputs.ref_suff:
            self.inputs.ref_suff = ''
        if not self.inputs.out_suff:
            self.inputs.out_suff = '_reoriented'
        
        ## Create output folder
        if not os.path.exists(self.inputs.out_dir):
            os.makedirs(self.inputs.out_dir)
        
        #print('in-dir: ', self.inputs.in_dir)
        infiles = Path(self.inputs.in_dir).glob('*' + self.inputs.in_suff + img_ext_type)
        
        for in_img_name in infiles:
            
            ## Get args
            in_bname = utils.get_basename(in_img_name, self.inputs.in_suff, [img_ext_type])
            if not self.inputs.ref_dir:
                ref_img_name = None
            else:
                ref_img_name = os.path.join(self.inputs.ref_dir, 
                                            in_bname + self.inputs.ref_suff + img_ext_type)
            out_img_name = os.path.join(self.inputs.out_dir,
                                        in_bname + self.inputs.out_suff + img_ext_type)
            
            ## Call the main function
            reorienter.apply_reorient(in_img_name, out_img_name, ref_img_name)

        # And we are done
        return runtime

    def _list_outputs(self):
        return {'out_dir': self.inputs.out_dir}
