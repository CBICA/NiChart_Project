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
    parser.add_argument('--data_file', type=str, help='Data file containing data frame.', default=None, required=False)

    args = parser.parse_args(sys.argv[1:])

    data_file = args.data_file

    app = QtWidgets.QApplication(sys.argv)
    
    with open('./style.qss', 'r') as f:
        style = f.read()
        # Set the current style sheet
    app.setStyleSheet(style)

    
    mw = MainWindow(dataFile=data_file)
    mw.show()

        #sys.exit(app.exec_())

if __name__ == '__main__':
    main()
