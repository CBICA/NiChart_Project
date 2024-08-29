# This Python file uses the following encoding: utf-8
"""
contact: software@cbica.upenn.edu
Copyright (c) 2018 University of Pennsylvania. All rights reserved.
Use of this source code is governed by license located in license file: https://github.com/CBICA/NiChart_Viewer/blob/main/LICENSE
"""
from PyQt5 import QtCore, QtWidgets

class CheckableQComboBox(QtWidgets.QComboBox):

    def __init__(self, parent=None):
        super(CheckableQComboBox, self).__init__(parent)

    # once there is a checkState set, it is rendered
    # here we assume default Unchecked
    def addItem(self, item):
        super(CheckableQComboBox, self).addItem(item)
        item = self.model().item(self.count()-1,0)
        item.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
        item.setCheckState(QtCore.Qt.Unchecked)

    def addItems(self, items):
        for i, item in enumerate(items):
            super(CheckableQComboBox, self).addItem(item)
            item = self.model().item(self.count()-1,0)
            item.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            item.setCheckState(QtCore.Qt.Unchecked)
            #item.setCheckState(QtCore.Qt.Checked)

    def itemChecked(self, index):
        item = self.model().item(index,0)
        return item.checkState() == QtCore.Qt.Checked

    def listCheckedItems(self):
        sel_list=[]
        for tmpInd in range(0, self.count()):
            if self.itemChecked(tmpInd):
                sel_list.append(self.itemText(tmpInd))
        return sel_list
                
        #item = self.model().item(i,0)
        #return item.checkState() == QtCore.Qt.Checked
