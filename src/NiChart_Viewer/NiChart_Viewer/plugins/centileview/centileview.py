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

class CentileView(QtWidgets.QWidget,BasePlugin):

    def __init__(self):
        super(CentileView,self).__init__()

        self.root_dir = None
        self.data_model_arr = None
        self.active_index = -1

        self.cmds = None

        root = os.path.dirname(__file__)

        self.readAdditionalInformation(root)
        self.ui = uic.loadUi(os.path.join(root, 'centileview.ui'),self)
        
        self.mdi = self.findChild(QMdiArea, 'mdiArea')       
        self.mdi.setBackground(QtGui.QColor(245,245,245,255))
        
        ## Panel for centile type
        self.ui.comboCentileType = QComboBox(self.ui)
        self.ui.comboCentileType.setEditable(False)
        self.ui.vlComboCentiles.addWidget(self.ui.comboCentileType)
        self.PopulateComboBox(self.ui.comboCentileType, ['Healthy Control',
                                                         'Healthy Control, Female',
                                                         'Healthy Control, Male',
                                                         'MCI',
                                                         'MCI, Female',
                                                         'MCI, Male',
                                                         'AD',
                                                         'AD, Female',
                                                         'AD, Male'], '--centile type--')
        
        ## Panel for Y var
        self.ui.comboYVar = QComboBox(self.ui)
        self.ui.comboYVar.setEditable(False)
        self.ui.vlComboY.addWidget(self.ui.comboYVar)
        
        ## Panel for Hue var
        self.ui.comboHueVar = QComboBox(self.ui)
        self.ui.comboHueVar.setEditable(False)
        self.ui.vlComboHue.addWidget(self.ui.comboHueVar)

        ## Options panel is not shown if there is no dataset loaded
        #self.ui.wOptions.hide()
        #self.ui.wVars.hide()
        #self.ui.wYVar.hide()
        #self.ui.wPlotBtn.hide()

        self.ui.edit_activeDset.setReadOnly(True)               

        self.ui.wOptions.setMaximumWidth(300)
    

    def SetupConnections(self):

        self.data_model_arr.active_dset_changed.connect(lambda: self.OnDataChanged())

        self.ui.comboHueVar.currentIndexChanged.connect(lambda: self.OnHueIndexChanged())

        self.ui.comboCentileType.currentIndexChanged.connect(self.OnCentileTypeChanged)
        self.ui.plotBtn.clicked.connect(lambda: self.OnPlotBtnClicked())

    def OnCentileTypeChanged(self):
        
        #root = os.path.dirname(__file__)
        
        logger.info('Centile ref data changed')

        dict_centile = {'Healthy Control' : 'NiChart_ALL_CN_Centiles',
                        'Healthy Control, Female' : 'NiChart_F_CN_Centiles',
                        'Healthy Control, Male' : 'NiChart_M_CN_Centiles',
                        'MCI' : 'NiChart_ALL_MCI_Centiles',
                        'MCI, Female' : 'NiChart_F_MCI_Centiles',
                        'MCI, Male' : 'NiChart_F_MCI_Centiles',
                        'AD' : 'NiChart_M_AD_Centiles',
                        'AD, Female' : 'NiChart_M_AD_Centiles',
                        'AD, Male' : 'NiChart_M_AD_Centiles'}

        self.selCentileType = self.ui.comboCentileType.currentText()
        self.selCentileFile = dict_centile[self.selCentileType]
        
        ## FIXME
        fCentile = os.path.join(self.root_dir, 'shared', 'centiles', 'NiChartcentiles', 
                                self.selCentileFile + '.csv')
        
        dio = DataIO()
        self.df_cent = dio.ReadCSVFile(fCentile)
        

        #self.statusbar.showMessage('Centile reference changed to: ' + self.selCentileType, 8000)
        self.statusbar.showMessage('Centile file: ' + fCentile, 8000)
            
    def OnPlotBtnClicked(self):

        dset_name = self.data_model_arr.dataset_names[self.active_index]        

        ## Get data
        df = self.data_model_arr.datasets[self.active_index].data

        ## Get user selections
        x_var = 'Age'

        hue_var = self.ui.comboHueVar.currentText()
        
        ## Prepare plot canvas  
        self.plotCanvas = PlotCanvas(self.ui)
        self.plotCanvas.axes = self.plotCanvas.fig.add_subplot(111)

        ## Read y var for reg plot
        y_var = self.ui.comboYVar.currentText()
        ## Plot data
        
        df_out = df
        
        self.statusbar.showMessage('Centile cols : ' + ','.join(self.df_cent.ROI_Name.unique()), 8000)

        #DataPlotScatter(self.plotCanvas.axes, df_out, x_var, y_var)
        DataPlotWithCentiles(self.plotCanvas.axes, df_out, x_var, y_var, self.df_cent, self.selCentileType, hue_var)

        self.plotCanvas.draw()

        ## Set data view to plot canvas
        sub = QMdiSubWindow()
        sub.setWidget(self.plotCanvas)
        self.mdi.addSubWindow(sub)        
        sub.show()
        self.mdi.tileSubWindows()
        
        ##-------
        ### Populate commands that will be written in a notebook
        #dset_name = self.data_model_arr.dataset_names[self.active_index]       

        ### Add function definitons to notebook
        #fCode = inspect.getsource(hue_regplot)
        #self.cmds.add_funcdef('hue_regplot', ['', fCode, ''])

        ##fCode = inspect.getsource(DataPlotScatter).replace('ax=axes','')
        #fCode = inspect.getsource(DataPlotScatter)
        #self.cmds.add_funcdef('DataPlotScatter', ['', fCode, ''])

        #fCode = inspect.getsource(DataPlotDist)
        #self.cmds.add_funcdef('DataPlotDist', ['', fCode, ''])


        ### Add cmds to call the function
        #cmds = ['']
        #cmds.append('# Plot data')
        #cmds.append('x_var = "' + x_var + '"')
        #if self.selPlotType == 'RegPlot':
            #cmds.append('y_var = "' + y_var + '"')
        #cmds.append('filterVar = "' + filterVar + '"')
        #str_filter_vals = '[' + ','.join('"{0}"'.format(x) for x in filterVals) + ']'
        #cmds.append('filterVals = ' + str_filter_vals)
        #cmds.append('hue_var = "' + hue_var + '"')
        #str_hue_vals = '[' + ','.join('"{0}"'.format(x) for x in hueVals) + ']'
        #cmds.append('hueVals = ' + str_hue_vals)
        #cmds.append('f, axes = plt.subplots(1, 1, figsize=(5, 4), dpi=100)')
        #if self.selPlotType == 'RegPlot':
            #cmds.append('axes = DataPlotScatter(axes, ' + dset_name + ', x_var, y_var, hue_var)')
        #if self.selPlotType == 'DistPlot':
            #cmds.append('axes = DataPlotDist(axes, ' + dset_name + ', x_var, hue_var)')
        #cmds.append('')
        #self.cmds.add_cmd(cmds)
        ##-------
        
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
            
    def OnHueIndexChanged(self):
        
        TH_NUM_UNIQ = 20
        
        selHue = self.ui.comboHueVar.currentText()
        #selHueVals = self.data_model_arr.datasets[self.active_index].data.unique()
        
        #if len(selHueVals) > TH_NUM_UNIQ:
            #self.errmsg.showMessage('Too many unique values for selection (' + 
                                    #str(len(selHueVals)) + '), skip')
            #return
        #self.PopulateComboBox(self.ui.comboHueVal, selHueVals)
        #self.ui.comboHueVal.show()
            

    def UpdatePanels(self, colNames):
        
        self.PopulateComboBox(self.ui.comboYVar, colNames, '--var name--')
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
        
