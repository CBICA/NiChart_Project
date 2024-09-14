import os
import re
from pathlib import Path
from typing import Union


def get_basename(
    in_file: Path, suffix_to_remove: str, ext_to_remove: list = [".nii.gz", ".nii"]
) -> Union[None, str]:
    """
    Get file basename
    - Extracts the base name from the input file
    - Removes a given suffix + file extension

    :param in_file: the input file
    :type in_file: str
    :param suffix_to_remove: passed suffix to be removed
    :type suffix_to_remove: str
    :param ext_to_remove: passed extensions to be removed.Default value:
                          ['.nii.gz', '.nii']
    :type ext_to_remove: list

    :return: the string without the suffix + file extension
    :rtype: str


    """
    # Get file basename
    out_str = os.path.basename(in_file)

    # Remove suffix and extension
    for tmp_ext in ext_to_remove:
        out_str, num_repl = re.subn(suffix_to_remove + tmp_ext + "$", "", out_str)
        if num_repl > 0:
            break

    if num_repl == 0:
        return out_str

    return out_str


def remove_common_suffix(list_files: list) -> list:
    """
    Detect common suffix to all images in the list and remove it to return a new list
    This list can be used as unique ids for input images
    (assumption: images have the same common suffix - example:  Subj1_T1_LPS.nii.gz -> Subj1)

    :param list_files: a list with all the filenames
    :type list_files: list

    :return: a list with the removed common suffix files
    :rtype: list

    """
    bnames = list_files
    if len(list_files) == 1:
        return bnames

    num_diff_suff = 1
    while num_diff_suff == 1:
        tmp_suff = [x[-1] for x in bnames]
        num_diff_suff = len(set(tmp_suff))
        if num_diff_suff == 1:
            bnames = [x[0:-1] for x in bnames]
    return bnames
