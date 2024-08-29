# This Python file uses the following encoding: utf-8
"""
contact: software@cbica.upenn.edu
Copyright (c) 2018 University of Pennsylvania. All rights reserved.
Use of this source code is governed by license located in license file: https://github.com/CBICA/NiChart_Viewer/blob/main/LICENSE
"""
import time
import pandas as pd
import numpy as np
import importlib.resources as pkg_resources
import sys
import json
import joblib
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5 import QtCore
from NiChart_Viewer.core import iStagingLogger

logger = iStagingLogger.get_logger(__name__)

class CmdModel(QObject):
    """This class holds a collection of commands to create a jupyter notebook file."""

    def __init__(self):

        QObject.__init__(self)
        """The constructor."""
        
        self.hdr_cmds = ['## NiChart_Viewer Notebook']
        self.hdr_cmds.append('#### Generated automatically by NiChart_Viewer')
        self.add_hdr()
        
        self.funcdef_cmds = ['#### Function definitions']
        self.funcnames = []
        
        self.cmds = ['#### Main notebook']


    def add_hdr(self):
        currTime = time.strftime('%H:%M%p %Z on %b %d, %Y')
        self.hdr_cmds.append('     ' + currTime)
        self.hdr_cmds.append('#### Import statements')
        self.hdr_cmds.append('')
        self.hdr_cmds.append('import pandas as pd')
        self.hdr_cmds.append('import numpy as np')
        self.hdr_cmds.append('import seaborn as sns')
        self.hdr_cmds.append('import matplotlib as mpl')
        self.hdr_cmds.append('import matplotlib.pyplot as plt')
        self.hdr_cmds.append('from matplotlib.cm import get_cmap')
        self.hdr_cmds.append('from matplotlib.lines import Line2D')
        self.hdr_cmds.append('import statsmodels.formula.api as sm')
        self.hdr_cmds.append('')
    
    def add_funcdef(self, fname, cvalue):

        if fname not in self.funcnames:
            self.funcnames.append(fname)
            self.funcdef_cmds = self.funcdef_cmds + cvalue

    def add_cmd(self, cvalue):
        self.cmds = self.cmds + cvalue

    def print_cmds(self):
        for i, tmpcmd in enumerate(self.hdr_cmds + self.funcdef_cmds + self.cmds):
            print('Cmd ' + str(i) + ' : ' + tmpcmd)
        
    ## Function to write commands as a notebook
    def cmds_to_notebook(self, fname):
    
        ## Components of a notebook initialized to empty
        d_code = {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": []
            }
        d_mark = {
                "cell_type": "markdown",
                "metadata": {},
                "source": []
            }
        d_meta = {
                "metadata": {
                    "kernelspec": {
                        "display_name": "Python 3",
                        "language": "python",
                        "name": "python3"
                    },
                    "language_info": {
                        "codemirror_mode": {
                            "name": "ipython",
                            "version": 3
                        },
                        "file_extension": ".py",
                        "mimetype": "text/x-python",
                        "name": "python",
                        "nbconvert_exporter": "python",
                        "pygments_lexer": "ipython3",
                        "version": "3"
                    }
                },
                "nbformat": 4,
                "nbformat_minor": 2
            }

        # Initialise variables
        notebook = {}       # Final notebook
        cells = []          # Cells of the notebook
        cell = []           # Contents of single cell

        # Read commands and store into cells
        cmd_type = 'end_block'        
        for i, tmp_cmd in enumerate(self.hdr_cmds + self.funcdef_cmds + self.cmds):
            
            ## Start of block
            if cmd_type == 'end_block':
                if tmp_cmd == "":
                    continue
                
                else:
                    cell.append("{}".format(tmp_cmd + '\n'))
                    if tmp_cmd.startswith('##'):
                        cmd_type = 'mark'
                    else:
                        cmd_type = 'code'

            ## Middle of block
            else:
                ## Block ends
                if tmp_cmd == "":
                    if cmd_type == 'mark':
                        d_mark["source"] = cell
                        cells.append(dict(d_mark)) 
                    
                    else:
                        d_code["source"] = cell
                        cells.append(dict(d_code)) 
                    cell = []
                    cmd_type = 'end_block'

                ## Block continues
                else:
                    cell.append("{}".format(tmp_cmd + '\n'))
                
        # Add to notebook
        notebook["cells"] = cells
        notebook.update(d_meta)

        # Write notebook
        with open(fname, "w", encoding="utf-8") as fp:
            json.dump(notebook, fp, indent=1, ensure_ascii=False)
