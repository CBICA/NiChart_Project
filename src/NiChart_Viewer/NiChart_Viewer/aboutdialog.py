# This Python file uses the following encoding: utf-8
"""
contact: software@cbica.upenn.edu
Copyright (c) 2018 University of Pennsylvania. All rights reserved.
Use of this source code is governed by license located in license file: https://github.com/CBICA/NiChart_Viewer/blob/main/LICENSE
"""

from PyQt5 import QtCore, QtWidgets, uic
import os
#from NiChart_Viewer.resources import resources

class AboutDialog(QtWidgets.QDialog):
    def __init__(self,parent=None):
        super(AboutDialog,self).__init__(parent)
        self.SetupUi()
        self.SetupConnections()

    def SetupConnections(self):
        self.buttonBox.accepted.connect(self.OnOkBtnClicked)

    def OnOkBtnClicked(self):
        self.hide()
 
    def SetupUi(self):
        root = os.path.dirname(__file__)
        self.ui = uic.loadUi(os.path.join(root, 'aboutdialog.ui'), self)
