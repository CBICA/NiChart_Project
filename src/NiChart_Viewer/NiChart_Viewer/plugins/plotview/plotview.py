from PyQt5.QtGui import *
from PyQt5 import QtGui, QtCore, QtWidgets, uic
from PyQt5.QtWidgets import QMdiArea, QMdiSubWindow, QTextEdit, QComboBox
import sys, os
import pandas as pd
from NiChart_Viewer.core.dataio import DataIO
# import dtale
from NiChart_Viewer.core.baseplugin import BasePlugin
from NiChart_Viewer.core import iStagingLogger
from NiChart_Viewer.core.gui.SearchableQComboBox import SearchableQComboBox
from NiChart_Viewer.core.gui.CheckableQComboBox import CheckableQComboBox
from NiChart_Viewer.core.plotcanvas import PlotCanvas
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D

#from NiChart_Viewer.core import datautils
from NiChart_Viewer.core.datautils import *

import inspect

logger = iStagingLogger.get_logger(__name__)

class PlotView(QtWidgets.QWidget,BasePlugin):

    def __init__(self):
        super(PlotView,self).__init__()

        self.data_model_arr = None
        self.active_index = -1

        self.cmds = None

        root = os.path.dirname(__file__)

        self.readAdditionalInformation(root)
        self.ui = uic.loadUi(os.path.join(root, 'plotview.ui'),self)
        
        self.mdi = self.findChild(QMdiArea, 'mdiArea')       
        self.mdi.setBackground(QtGui.QColor(245,245,245,255))
        
        ## Panel for action
        self.ui.comboPlotType = QComboBox(self.ui)
        self.ui.comboPlotType.setEditable(False)
        self.ui.vlPlotType.addWidget(self.ui.comboPlotType)
        self.PopulateComboBox(self.ui.comboPlotType, ['DistPlot', 'RegPlot'], '--plot type--')        
        
        ## Panel for X var
        self.ui.comboXVar = QComboBox(self.ui)
        self.ui.comboXVar.setEditable(False)
        self.ui.vlComboX.addWidget(self.ui.comboXVar)

        ## Panel for Y var
        self.ui.comboYVar = QComboBox(self.ui)
        self.ui.comboYVar.setEditable(False)
        self.ui.vlComboY.addWidget(self.ui.comboYVar)
        
        ## Panel for filter var
        self.ui.comboFilterVar = QComboBox(self.ui)
        self.ui.comboFilterVar.setEditable(False)
        self.ui.vlComboFilter.addWidget(self.ui.comboFilterVar)
        
        self.ui.comboFilterVal = CheckableQComboBox(self.ui)
        self.ui.comboFilterVal.setEditable(False)
        self.ui.hlComboFilterVal.addWidget(self.ui.comboFilterVal)

        self.ui.label_filter.hide()
        #self.ui.box_filter.hide()
        self.ui.comboFilterVar.hide()
        self.ui.comboFilterVal.hide()

        ## Panel for hue var
        self.ui.comboHueVar = QComboBox(self.ui)
        self.ui.comboHueVar.setEditable(False)
        self.ui.vlComboHue.addWidget(self.ui.comboHueVar)

        self.ui.comboHueVal = CheckableQComboBox(self.ui)
        self.ui.comboHueVal.setEditable(False)
        self.ui.hlComboHueVal.addWidget(self.ui.comboHueVal)

        self.ui.comboHueVal.hide()
        
        ## Options panel is not shown if there is no dataset loaded
        self.ui.wOptions.hide()
        self.ui.wVars.hide()
        self.ui.wYVar.hide()
        self.ui.wPlotBtn.hide()

        self.ui.edit_activeDset.setReadOnly(True)               

        self.ui.wOptions.setMaximumWidth(300)
    

    def SetupConnections(self):

        self.data_model_arr.active_dset_changed.connect(lambda: self.OnDataChanged())

        self.ui.comboHueVar.currentIndexChanged.connect(lambda: self.OnHueIndexChanged())
        self.ui.comboFilterVar.currentIndexChanged.connect(lambda: self.OnFilterIndexChanged())

        self.ui.comboPlotType.currentIndexChanged.connect(self.OnPlotTypeChanged)
        self.ui.plotBtn.clicked.connect(lambda: self.OnPlotBtnClicked())

    def OnPlotTypeChanged(self):
        
        logger.info('Plot type selection changed')

        self.selPlotType = self.ui.comboPlotType.currentText()
        
        if self.selPlotType == 'RegPlot':
            self.ui.wVars.show()
            self.ui.wYVar.show()
            self.ui.wPlotBtn.show()
            

        if self.selPlotType == 'DistPlot':
            self.ui.wVars.show()
            self.ui.wYVar.hide()
            self.ui.wPlotBtn.show()

        self.statusbar.showMessage('Plot type changed to: ' + self.selPlotType, 8000)

    def OnFilterIndexChanged(self):
        
        ## Threshold to show categorical values for selection
        TH_NUM_UNIQ = 20    
        
        selFilter = self.ui.comboFilterVar.currentText()
        selFilterVals = self.data_model_arr.datasets[self.active_index].data[selFilter].unique()
        
        if len(selFilterVals) > TH_NUM_UNIQ:
            self.errmsg.showMessage('Too many unique values for selection (' + 
                                    str(len(selFilterVals)) + '), skip')
            return;
        self.PopulateComboBox(self.ui.comboFilterVal, selFilterVals)
        #self.ui.comboFilterVal.show()

    def OnHueIndexChanged(self):
        
        TH_NUM_UNIQ = 20
        
        selHue = self.ui.comboHueVar.currentText()
        selHueVals = self.data_model_arr.datasets[self.active_index].data[selHue].unique()
        
        if len(selHueVals) > TH_NUM_UNIQ:
            self.errmsg.showMessage('Too many unique values for selection (' + 
                                    str(len(selHueVals)) + '), skip')
            return
        self.PopulateComboBox(self.ui.comboHueVal, selHueVals)
        self.ui.comboHueVal.show()
            
    def OnPlotBtnClicked(self):

        dset_name = self.data_model_arr.dataset_names[self.active_index]        

        ## Get data
        df = self.data_model_arr.datasets[self.active_index].data

        ## Get user selections
        x_var = self.ui.comboXVar.currentText()
        hue_var = self.ui.comboHueVar.currentText()
        hueVals = self.ui.comboHueVal.listCheckedItems()
        filterVar = self.ui.comboFilterVar.currentText()
        filterVals = self.ui.comboFilterVal.listCheckedItems()
        if (filterVar == '--var name--') | (filterVals == []):
            filterVar = ''
        if (hue_var == '--var name--') | (hueVals == []):
            hue_var = ''

        ## Filter data
        df_out = df
        res_tmp = DataFilter(df, filterVar, filterVals)
        if res_tmp['out_code'] == 0:
            df_out = res_tmp['df_out']
        #res_tmp = DataFilter(df_out, hue_var, hueVals)
        #if res_tmp['out_code'] == 0:
            #df_out = res_tmp['df_out']

        ## Prepare plot canvas  
        self.plotCanvas = PlotCanvas(self.ui)
        self.plotCanvas.axes = self.plotCanvas.fig.add_subplot(111)
        if self.selPlotType == 'RegPlot':
            ## Read y var for reg plot
            y_var = self.ui.comboYVar.currentText()
            ## Plot data
            DataPlotScatter(self.plotCanvas.axes, df_out, x_var, y_var, hue_var)

        if self.selPlotType == 'DistPlot':
            ## Plot data
            DataPlotDist(self.plotCanvas.axes, df_out, x_var, hue_var)
        self.plotCanvas.draw()

        ## Set data view to plot canvas
        sub = QMdiSubWindow()
        sub.setWidget(self.plotCanvas)
        self.mdi.addSubWindow(sub)        
        sub.show()
        self.mdi.tileSubWindows()
        
        ##-------
        ## Populate commands that will be written in a notebook
        dset_name = self.data_model_arr.dataset_names[self.active_index]       

        ## Add function definitons to notebook
        fCode = inspect.getsource(hue_regplot)
        self.cmds.add_funcdef('hue_regplot', ['', fCode, ''])

        #fCode = inspect.getsource(DataPlotScatter).replace('ax=axes','')
        fCode = inspect.getsource(DataPlotScatter)
        self.cmds.add_funcdef('DataPlotScatter', ['', fCode, ''])

        fCode = inspect.getsource(DataPlotDist)
        self.cmds.add_funcdef('DataPlotDist', ['', fCode, ''])


        ## Add cmds to call the function
        cmds = ['']
        cmds.append('# Plot data')
        cmds.append('x_var = "' + x_var + '"')
        if self.selPlotType == 'RegPlot':
            cmds.append('y_var = "' + y_var + '"')
        cmds.append('filterVar = "' + filterVar + '"')
        str_filter_vals = '[' + ','.join('"{0}"'.format(x) for x in filterVals) + ']'
        cmds.append('filterVals = ' + str_filter_vals)
        cmds.append('hue_var = "' + hue_var + '"')
        str_hue_vals = '[' + ','.join('"{0}"'.format(x) for x in hueVals) + ']'
        cmds.append('hueVals = ' + str_hue_vals)
        cmds.append('f, axes = plt.subplots(1, 1, figsize=(5, 4), dpi=100)')
        if self.selPlotType == 'RegPlot':
            cmds.append('axes = DataPlotScatter(axes, ' + dset_name + ', x_var, y_var, hue_var)')
        if self.selPlotType == 'DistPlot':
            cmds.append('axes = DataPlotDist(axes, ' + dset_name + ', x_var, hue_var)')
        cmds.append('')
        self.cmds.add_cmd(cmds)
        #-------
        
    def OnDataChanged(self):

        if self.data_model_arr.active_index >= 0:
     
            ## Make options panel visible
            self.ui.wOptions.show()
        
            ## Set fields for various options     
            self.active_index = self.data_model_arr.active_index
                
            ## Get data variables
            dataset = self.data_model_arr.datasets[self.active_index]
            colNames = dataset.data.columns.tolist()

            logger.info(self.active_index)
            
            ## Set active dset name
            self.ui.edit_activeDset.setText(self.data_model_arr.dataset_names[self.active_index])

            ## Update selection, sorting and drop duplicates panels
            self.UpdatePanels(colNames)

    def UpdatePanels(self, colNames):
        
        self.PopulateComboBox(self.ui.comboXVar, colNames, '--var name--')
        self.PopulateComboBox(self.ui.comboYVar, colNames, '--var name--')
        self.PopulateComboBox(self.ui.comboFilterVar, colNames, '--var name--')
        self.PopulateComboBox(self.ui.comboHueVar, colNames, '--var name--')
        
    # Add the values to comboBox
    def PopulateComboBox(self, cbox, values, strPlaceholder = None, bypassCheckable=False):
        cbox.blockSignals(True)
        cbox.clear()

        if bypassCheckable:
            cbox.addItemsNotCheckable(values)  ## The checkableqcombo for var categories
                                               ##   should not be checkable
        else:
            cbox.addItems(values)
            
        if strPlaceholder is not None:
            cbox.setCurrentIndex(-1)
            cbox.setEditable(True)
            cbox.setCurrentText(strPlaceholder)
        cbox.blockSignals(False)
        
