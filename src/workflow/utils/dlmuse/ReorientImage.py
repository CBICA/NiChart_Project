import nibabel as nib
from nibabel.orientations import axcodes2ornt, ornt_transform

def reorient_img(in_img, ref_orient = 'LPS', out_img):
    """
    Reorient input image to reference orientation
    """
    # Read input img
    nii_in = nib.load(in_img)

    # Find transform from current (approximate) orientation to
    # target, in nibabel orientation matrix and affine forms
    orient_in = nib.io_orientation(nii_in.affine)
    orient_out = axcodes2ornt(ref_orient)
    transform = ornt_transform(orient_in, orient_out)
    # affine_xfm = inv_ornt_aff(transform, nii_in.shape)

    # Apply transform
    reoriented = nii_in.as_reoriented(transform)

    # Write to out file
    reoriented.to_filename(out_img)
