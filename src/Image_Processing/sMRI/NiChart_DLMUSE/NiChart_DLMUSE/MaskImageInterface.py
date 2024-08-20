import os
from pathlib import Path
from typing import Any

from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    Directory,
    File,
    TraitedSpec,
    traits,
)

from NiChart_DLMUSE import MaskImage as masker
from NiChart_DLMUSE import utils


class MaskImageInputSpec(BaseInterfaceInputSpec):
    in_dir = Directory(mandatory=True, desc="the input dir")
    in_suff = traits.Str(mandatory=False, desc="the input image suffix")
    mask_dir = Directory(mandatory=True, desc="the mask img directory")
    mask_suff = traits.Str(mandatory=False, desc="the mask image suffix")
    out_dir = Directory(mandatory=True, desc="the output dir")
    out_suff = traits.Str(mandatory=False, desc="the out image suffix")


class MaskImageOutputSpec(TraitedSpec):
    out_dir = File(desc="the output image")


class MaskImage(BaseInterface):
    input_spec = MaskImageInputSpec
    output_spec = MaskImageOutputSpec

    def _run_interface(self, runtime: Any) -> Any:
        img_ext_type = ".nii.gz"

        # Set input args
        if not self.inputs.in_suff:
            self.inputs.in_suff = ""
        if not self.inputs.mask_suff:
            self.inputs.mask_suff = ""
        if not self.inputs.out_suff:
            self.inputs.out_suff = "_masked"

        # Create output folder
        if not os.path.exists(self.inputs.out_dir):
            os.makedirs(self.inputs.out_dir)

        infiles = Path(self.inputs.in_dir).glob(
            "*" + self.inputs.in_suff + img_ext_type
        )

        for in_img_name in infiles:

            # Get args
            in_bname = utils.get_basename(
                in_img_name, self.inputs.in_suff, [img_ext_type]
            )
            mask_img_name = os.path.join(
                self.inputs.mask_dir, in_bname + self.inputs.mask_suff + img_ext_type
            )
            out_img_name = os.path.join(
                self.inputs.out_dir, in_bname + self.inputs.out_suff + img_ext_type
            )

            # Call the main function
            masker.apply_mask(in_img_name, mask_img_name, out_img_name)

        # And we are done
        return runtime

    def _list_outputs(self) -> dict:
        return {"out_dir": self.inputs.out_dir}
