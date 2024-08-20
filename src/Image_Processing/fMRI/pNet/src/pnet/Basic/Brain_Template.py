# Yuncong Ma, 11/30/2023
# Brain template information and data
# Brain template is a dictionary containing 'Data_Type' and 'Data_Format'
# 'Data_Type' is 'Surface', 'Volume', 'Surface-Volume'
# Brain template is categorized into surface and volume types
# If data type is volume, 'Brain_Mask' and 'Overlay_Image' are required, and both are 3D images
# If data type is surface, 'Shape' and 'Brain_Mask' are required, and 'Shape_Inflated' is optional
# 'Shape' and 'Shape_Inflated' are nested dictionaries containing 'L' and 'R', with 'faces' and 'vertices' in both
# 'faces' contains [N 3] matrix, and 'vertices' contains [N 3] matrix
# 'Brain_Mask' is also a dictionary containing 'L' and 'R', which store 1D vectors

#########################################
# Packages
import os

#########################################

# Get the directory of pNet based the location of this file
current_file_path = os.path.abspath(__file__)
#dir_python_package = os.path.dirname(current_file_path)
current_dir = os.path.dirname(current_file_path)    # mod by hm
#dir_python_package = os.path.dirname(current_dir)   # mod by hm
#dir_pNet = os.path.dirname(dir_python_package)
dir_pNet = os.path.dirname(current_dir)
#########################################
dir_Template = os.path.join(dir_pNet, 'Brain_Template')
# HCP surface
file_HCP_surf = os.path.join(dir_Template, 'HCP_Surface', 'Brain_Template.json.zip')

# HCP surface-volume
file_HCP_surf_vol = os.path.join(dir_Template, 'HCP_Surface_Volume', 'Brain_Template.json.zip')

# HCP volume
file_HCP_vol = os.path.join(dir_Template, 'HCP_Volume', 'Brain_Template.json.zip')

# FreeSurfer surface
file_FS_surf = os.path.join(dir_Template, 'FreeSurfer_fsaverage5', 'Brain_Template.json.zip')

# MNI volume
file_MNI_vol = os.path.join(dir_Template, 'MNI_Volume', 'Brain_Template.json.zip')
# Organize example into a class variable


class Brain_Template:

    # HCP surface
    file_HCP_surf = file_HCP_surf

    # HCP surface-volume
    file_HCP_surf_vol = file_HCP_surf_vol

    # HCP volume
    file_HCP_vol = file_HCP_vol

    # FreeSurfer surface
    file_FS_surf = file_FS_surf

    # MNI volume
    file_MNI_vol = file_MNI_vol
