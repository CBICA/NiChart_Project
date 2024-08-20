# This Python file uses the following encoding: utf-8
"""
contact: software@cbica.upenn.edu
Copyright (c) 2018 University of Pennsylvania. All rights reserved.
Use of this source code is governed by license located in license file: https://github.com/CBICA/NiChart_Viewer/blob/main/LICENSE
"""
from PyQt5 import QtCore, QtWidgets

class CheckableQComboBox(QtWidgets.QComboBox):


    ##https://morioh.com/p/d1e70112347c

    def __init__(self, parent=None):
        super(CheckableQComboBox, self).__init__(parent)

        self.listItemText = [] 

        #self.view().pressed.connect(self.handle_item_pressed)

    # once there is a checkState set, it is rendered
    # here we assume default Unchecked
    def addItem(self, item):
        super(CheckableQComboBox, self).addItem(item)
        item = self.model().item(self.count()-1,0)
        item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
        item.setCheckState(QtCore.Qt.Unchecked)

    def addItems(self, items):
        for i, item in enumerate(items):
            super(CheckableQComboBox, self).addItem(str(item))
            item = self.model().item(self.count()-1,0)
            item.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            item.setCheckState(QtCore.Qt.Unchecked)
            #item.setCheckState(QtCore.Qt.Checked)
        self.listItemText = items

    def addItemsNotCheckable(self, items):
        for i, item in enumerate(items):
            super(CheckableQComboBox, self).addItem(item)
            item = self.model().item(self.count()-1,0)
            item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            item.setCheckState(QtCore.Qt.Unchecked)
            #item.setCheckState(QtCore.Qt.Checked)
        self.listItemText = items

    def itemChecked(self, index):
        item = self.model().item(index,0)
        return item.checkState() == QtCore.Qt.Checked

    def uncheckItems(self, listItems):
        indSel = [i for i, e in enumerate(self.listItemText) if e in set(listItems)]

        for i, tmpInd in enumerate(indSel):
            item = self.model().item(tmpInd,0)
            item.setCheckState(QtCore.Qt.Unchecked)

    def checkAllItems(self):
        for i, tmpInd in enumerate(self.listItemText):
            item = self.model().item(i,0)
            item.setCheckState(QtCore.Qt.Checked)

    def checkItems(self, listItems):
        indSel = [i for i, e in enumerate(self.listItemText) if e in set(listItems)]

        for i, tmpInd in enumerate(indSel):
            item = self.model().item(tmpInd,0)
            item.setCheckState(QtCore.Qt.Checked)

    def listCheckedItems(self):
        sel_list=[]
        for tmpInd in range(0, self.count()):
            if self.itemChecked(tmpInd):
                sel_list.append(self.itemText(tmpInd))
        return sel_list
                
        #item = self.model().item(i,0)
        #return item.checkState() == QtCore.Qt.Checked

    def handle_item_pressed(self, index):

        item = self.model().itemFromIndex(index)
  
        ## make it check if unchecked and vice-versa
        #if item.checkState() == Qt.Checked:
            #item.setCheckState(Qt.Unchecked)
        #else:
            #item.setCheckState(Qt.Checked)

        #print(self.itemText(index))
        print(item.text())
  
        ## calling method
        #self.check_items()
