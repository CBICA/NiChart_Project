from pathlib import Path
from typing import Union

import nibabel as nib
from nibabel.orientations import axcodes2ornt, ornt_transform


def apply_reorient(
    in_img_name: Path, out_img_name: Path, ref_img_name: Union[str, None] = None
) -> None:
    """
    Reorient input image to referenced image orientation(if provided), otherwise, to LPS

    :param in_img_name: the passed image
    :type in_img_name: str
    :param out_img_name: the wanted output filename
    :type out_img_name: str
    :param ref_img_name: the passed ref image orientation. Default value = None
    :type ref_img_name: str
    """
    # Read input img
    nii_in = nib.load(in_img_name)

    # Detect target orient
    if ref_img_name is None:
        ref_orient = "LPS"
    else:
        nii_ref = nib.load(ref_img_name)
        ref_orient = nib.aff2axcodes(nii_ref.affine)
        ref_orient = "".join(ref_orient)

    # Find transform from current (approximate) orientation to
    # target, in nibabel orientation matrix and affine forms
    orient_in = nib.io_orientation(nii_in.affine)
    orient_out = axcodes2ornt(ref_orient)
    transform = ornt_transform(orient_in, orient_out)
    # affine_xfm = inv_ornt_aff(transform, nii_in.shape)

    # Apply transform
    reoriented = nii_in.as_reoriented(transform)

    # Write to out file
    reoriented.to_filename(out_img_name)
