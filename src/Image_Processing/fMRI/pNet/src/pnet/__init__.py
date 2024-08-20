# Yuncong Ma, 1/19/2024
# pNet
# This script provides the highest level organization of pNet
# It provides workflows of pNet, and examples
# It includes five modules of pNet, underlying functions

#########################################
# Packages
import os
import sys

# path of pNet
current_file_path = os.path.abspath(__file__)
dir_python = os.path.dirname(current_file_path)
dir_pNet = os.path.dirname(dir_python)
dir_brain_template = os.path.join(dir_pNet, 'Brain_Template')
dir_example = os.path.join(dir_pNet, 'Example')

sys.path.append(dir_python)

# Example

# Brain templates

# Module
# This script builds the five modules of pNet
# Functions for modules of pNet
from Module.Data_Input import *
from Module.FN_Computation_torch import *
from Module.FN_Computation import *
from Module.Visualization import *
from Module.Quality_Control import *
from Module.FN_Computation_torch import *
from Report.Web_Report import *
from Workflow.Workflow_Func import *


