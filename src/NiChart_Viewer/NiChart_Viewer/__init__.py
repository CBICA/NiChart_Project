# This Python file uses the following encoding: utf-8
"""
contact: software@cbica.upenn.edu
Copyright (c) 2018 University of Pennsylvania. All rights reserved.
Use of this source code is governed by license located in license file: https://github.com/CBICA/NiChart_Viewer/blob/main/LICENSE
"""

from PyQt5 import QtCore, QtGui, QtWidgets
import argparse
import os, sys
from NiChart_Viewer.mainwindow import MainWindow
from NiChart_Viewer.NiChart_Viewer_CmdApp import NiChart_Viewer_CmdApp

def main():
    parser = argparse.ArgumentParser(description='NiChart_Viewer Data Visualization and Preparation')
    parser.add_argument('--data_file', type=str, help='Data file containing data frame. Users can use --data_file multiple times to load additional data files', default=None, required=False, action='append')
    #parser.add_argument('--dict_file', type=str, help='Dict file containing data dictionary. Users can use --data_dict multiple times to load additional data dictionaries', default=None, required=False, action='append')
    #parser.add_argument("-nogui", action="store_true", help="Launch application in CLI mode to do data processing without any visualization or graphical user interface.")

    args = parser.parse_args(sys.argv[1:])

    data_files = args.data_file
    #dict_files = args.dict_file
    #noGUI = args.nogui

    #if(noGUI):
        #app = QtCore.QCoreApplication(sys.argv)
        
    #else:
    app = QtWidgets.QApplication(sys.argv)
    
    # Set the style sheet
    styleFile = os.path.join(os.path.dirname(__file__), 'shared', 'styles',
                                'NiChart_Viewer_Stylesheet.qss')
    with open(styleFile, 'r') as f:         ## FIXME this is absolute path to curr dir
        style = f.read()
    app.setStyleSheet(style)
    
    #mw = MainWindow(dataFiles = data_files, dictFiles = dict_files)
    mw = MainWindow(dataFiles = data_files)
    mw.show()

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
