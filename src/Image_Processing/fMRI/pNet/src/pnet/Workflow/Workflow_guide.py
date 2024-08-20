# Yuncong Ma, 10/2/2023
import os
import sys
parent_directory = os.path.abspath('..')
sys.path.append(parent_directory)

import pnet

# This is to run a step-by-step workflow setup in terminal
if __name__ == '__main__':
    pnet.workflow_guide()
