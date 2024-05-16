import os
import re

def get_basename(in_file, suffix_to_remove, ext_to_remove = ['.nii.gz', '.nii']):
    '''Get file basename 
    - Extracts the base name from the input file
    - Removes a given suffix + file extension
    '''
    ## Get file basename
    out_str = os.path.basename(in_file)

    ## Remove suffix and extension
    for tmp_ext in ext_to_remove:
        out_str, num_repl = re.subn(suffix_to_remove + tmp_ext + '$', '', out_str)
        if num_repl > 0:
            break

    ## Return basename
    if num_repl == 0:
        return None
    return out_str


def remove_common_suffix(list_files):
    '''Detect common suffix to all images in the list and remove it to return a new list
       This list can be used as unique ids for input images 
       (assumption: images have the same common suffix - example:  Subj1_T1_LPS.nii.gz -> Subj1)
    '''
    bnames = list_files    
    if len(list_files) == 1:
        return bnames
    
    common_suff = ''
    num_diff_suff = 1
    while num_diff_suff == 1:
        tmp_suff = [x[-1] for x in bnames]
        num_diff_suff = len(set(tmp_suff))
        if num_diff_suff == 1:
            bnames = [x[0:-1] for x in bnames]
    return bnames