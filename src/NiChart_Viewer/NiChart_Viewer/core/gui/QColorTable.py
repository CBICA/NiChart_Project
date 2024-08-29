# This Python file uses the following encoding: utf-8
"""
Author: Guray Erus
contact: software@cbica.upenn.edu
Copyright (c) 2018 University of Pennsylvania. All rights reserved.
Use of this source code is governed by license located in license file: https://github.com/CBICA/NiChart_Viewer/blob/main/LICENSE
"""

from PyQt5 import QtCore, QtWidgets, QtGui
from NiChart_Viewer.core.gui.CheckableQComboBox import CheckableQComboBox
from NiChart_Viewer.core import iStagingLogger
import numpy as np

logger = iStagingLogger.get_logger(__name__)

class QColorTable(QtGui.QTableWidget):
    def __init__(self, thestruct, *args):
        QtGui.QTableWidget.__init__(self, *args)
        self.data = thestruct
        self.setmydata()

    def setmydata(self):
        n = 0
        for key in self.data:
            m = 0
            for item in self.data[key]:
                newitem = QtGui.QTableWidgetItem(item)
                if key == 'A':
                    newitem.setBackground(QtGui.QColor(100,100,150))
                elif key == 'B':
                    newitem.setBackground(QtGui.QColor(100,150,100))
                else:
                    newitem.setBackground(QtGui.QColor(150,100,100))
                self.setItem(m, n, newitem)
                m += 1
            n += 1
