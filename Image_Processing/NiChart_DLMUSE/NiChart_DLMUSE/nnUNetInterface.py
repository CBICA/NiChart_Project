from nipype.interfaces.base import (CommandLine, CommandLineInputSpec,
                                    TraitedSpec, traits, Directory)
import os
###--------Interface--------
class nnUNetInferenceInputSpec(CommandLineInputSpec):
    in_dir = Directory(mandatory=True,
                       argstr='-i %s', 
                       position=0,
                       desc='the input folder')
    out_dir = Directory(mandatory=True,
                        argstr='-o %s', 
                        position=-1, 
                        desc='the output folder')
    f_val = traits.Int(argstr='-f %d',
                        desc="f val: default 0")
    t_val = traits.Int(argstr='-t %d',
                        desc="t val: default 803")
    m_val = traits.Str(argstr='-m %s',
                       desc="m val: default 3d_fullres")
    tr_val = traits.Str(argstr='-tr %s',
                        desc="tr val: default nnUNetTrainerV2")
    disable_tta = traits.Bool(argstr='--disable_tta',
                              desc="disable tta: default False")
    all_in_gpu = traits.Str(argstr='--all_in_gpu %s',
                                desc="all in gpu: 'True', 'False' or 'None'. Default 'None'")
    mode = traits.Str(argstr='--mode %s',
                        desc="mode: default normal")

class nnUNetInferenceOutputSpec(TraitedSpec):
    out_dir = Directory(desc='the output folder')
    
class nnUNetInference(CommandLine):
        _cmd = 'nnUNet_predict'
        input_spec = nnUNetInferenceInputSpec
        output_spec = nnUNetInferenceOutputSpec

        def _list_outputs(self):
            outputs = self.output_spec().get()
            outputs['out_dir'] = self.inputs.out_dir
            return outputs

