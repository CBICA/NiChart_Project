# This Python file uses the following encoding: utf-8
"""
contact: software@cbica.upenn.edu
Copyright (c) 2018 University of Pennsylvania. All rights reserved.
Use of this source code is governed by license located in license file: https://github.com/CBICA/NiChart_Viewer/blob/main/LICENSE
"""

from PyQt5 import QtCore, QtWidgets
from NiChart_Viewer.core.gui.CheckableQComboBox import CheckableQComboBox
from NiChart_Viewer.core import iStagingLogger
import numpy as np

logger = iStagingLogger.get_logger(__name__)

class NestedQMenu(QtWidgets.QMenu):

    def __init__(self, parent=None):
        super(NestedQMenu, self).__init__(parent)

    # once there is a checkState set, it is rendered
    # here we assume default Unchecked


    def addNestedItems(self, df):
        self.clear()
        
        if df is not None:
        
            varCats = df.index.unique().tolist()
            
            logger.info('XXXXXXXXXx')
            logger.info(varCats)
            logger.info(df)
            
            for varCat in varCats:

                logger.info('HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH')
                logger.info(varCat)
                

                varNames = df.loc[varCat].VarName.tolist()

                logger.info(varNames)


                self.projCombo = CheckableQComboBox(self)
                self.projCombo.addItems(varNames)
                
                self.comboList = QtWidgets.QWidgetAction(None)
                self.comboList.setDefaultWidget(self.projCombo)
                
                self.popupMenu = QtWidgets.QMenu(self)
                self.popupMenu.setTitle(varCat)
                self.popupMenu.addAction(self.comboList)
                
                super(NestedQMenu, self).addMenu(self.popupMenu)
            

    def addSubMenuItem(self):
        
        self.projCombo = CheckableQComboBox(self)
        self.projCombo.addItem('Item1')
        self.projCombo.addItem('Item2')
        self.projCombo.addItem('Item3')
        
        self.comboList = QtWidgets.QWidgetAction(None)
        self.comboList.setDefaultWidget(self.projCombo)
        
        self.popupMenu2A = QtWidgets.QMenu(self)
        
        self.popupMenu2A.setTitle("Shape Data")
        self.popupMenu2A.addAction(self.comboList)
        
        super(NestedQMenu, self).addMenu(self.popupMenu2A)

    def addMenuItem(self, item):
        super(QMenu, self).addMenu(item)

    def addMenuItems(self, items):
        for i, item in enumerate(items):
            super(QMenu, self).addItem(item)

