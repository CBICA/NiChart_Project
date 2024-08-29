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
import statsmodels.formula.api as sm
from NiChart_Viewer.core.model.datamodel import DataModel, DataModelArr, PandasModel

from NiChart_Viewer.core.datautils import *

import inspect

logger = iStagingLogger.get_logger(__name__)

class IcvCorrView(QtWidgets.QWidget,BasePlugin):

    def __init__(self):
        super(IcvCorrView,self).__init__()

        self.data_model_arr = None
        self.active_index = -1
        
        self.cmds = None

        self.TH_NUM_UNIQ = 20

        root = os.path.dirname(__file__)

        self.readAdditionalInformation(root)
        self.ui = uic.loadUi(os.path.join(root, 'icvcorrview.ui'),self)
        
        self.mdi = self.findChild(QMdiArea, 'mdiArea')       
        self.mdi.setBackground(QtGui.QColor(245,245,245,255))
        
        ## Panel for norm var
        self.ui.comboNormVar = QComboBox(self.ui)
        self.ui.comboNormVar.setEditable(False)
        self.ui.vlComboNormVar.addWidget(self.ui.comboNormVar)
        
        ## Options panel is not shown if there is no dataset loaded
        #self.ui.wNormVars.hide()
        #self.ui.wAdjustVars.hide()
        #self.ui.wOutVars.hide()

        ## Default value in adj cov view is to create new dset (not to overwrite the active dset)
        #self.ui.check_createnew.hide()        
        #self.ui.check_createnew.setCheckState(QtCore.Qt.Checked)

        self.ui.edit_activeDset.setReadOnly(True)
        
        self.ui.edit_out_suff.setText('PercICV')
        
        self.ui.wOptions.setMaximumWidth(300)
        

    def SetupConnections(self):
        
        self.data_model_arr.active_dset_changed.connect(lambda: self.OnDataChanged())

        #self.ui.comboSelVar.currentIndexChanged.connect(lambda: self.OnSelIndexChanged())

        self.ui.normalizeBtn.clicked.connect(lambda: self.OnNormalizeBtnClicked())

    def OnNormalizeBtnClicked(self):
        '''Function to normalize data
        '''
        ## Get data
        df = self.data_model_arr.datasets[self.active_index].data
        dset_name = self.data_model_arr.dataset_names[self.active_index]        

        ## Get user selections
        norm_var = self.ui.comboNormVar.currentText()        
        out_suff = self.ui.edit_out_suff.text()
        
        ## Calculate results
        res_tmp = DataPercICV(df, norm_var, out_suff)
        if res_tmp['out_code'] != 0:
            self.errmsg.showMessage(res_tmp['out_msg'])
            return;
        df_out = res_tmp['df_out']
        out_vars = res_tmp['out_vars']

        ## Create new dataset or update current active dataset
        #if self.ui.check_createnew.isChecked():
            #dmodel = DataModel(df_out, dset_name + '_Normalized')
            #self.data_model_arr.AddDataset(dmodel)
            #self.data_model_arr.OnDataChanged()

        #else:
            #self.data_model_arr.datasets[self.active_index].data = df_out

        self.data_model_arr.datasets[self.active_index].data = df_out
            
        ## Call signal for change in data
        self.data_model_arr.OnDataChanged()        
        
        ## Display the table
#        self.statusbar.showMessage('Dataframe updated, size: ' + str(df_out.shape), 8000)          
        self.statusbar.showMessage('Dataframe updated: normalized values are added (columns with suffix: _' 
            + out_suff + '), data size: ' + str(df_out.shape), 10000)
        WidgetShowTable(self)
        
        ##-------
        ## Populate commands that will be written in a notebook

        ## Add NormalizeData function definiton to notebook
        fCode = inspect.getsource(DataNormalize).replace('(self, ','(')
        self.cmds.add_funcdef('NormalizeData', ['', fCode, ''])
        
        ## Add cmds to call the function
        cmds = ['']
        cmds.append('# Normalize data')

        str_out_vars = '[' + ','.join('"{0}"'.format(x) for x in out_vars) + ']'
        cmds.append('out_vars = ' + str_out_vars)

        cmds.append('norm_var = "' + norm_var + '"')
        
        cmds.append('out_suff  = "' + out_suff + '"')
        
        cmds.append(dset_name + ', outVarNames = NormalizeData(' + dset_name + ', out_vars, norm_var, out_suff)')
        
        cmds.append(dset_name + '[outVarNames].head()')
        cmds.append('')
        self.cmds.add_cmd(cmds)
        

        
    def ShowTable(self, df = None, dset_name = None):

        ## Read data and user selection
        if df is None:
            dset_name = self.data_model_arr.dataset_names[self.active_index]
            #dset_fname = self.data_model_arr.datasets[self.active_index].file_name
            df = self.data_model_arr.datasets[self.active_index].data
            
        ## Load data to data view 
        self.dataView = QtWidgets.QTableView()
        
        ## Reduce data size to make the app run faster
        df_tmp = df.head(self.data_model_arr.TABLE_MAXROWS)

        ## Round values for display
        df_tmp = df_tmp.applymap(lambda x: round(x, 2) if isinstance(x, (float, int)) else x)

        self.PopulateTable(df_tmp)

        ## Set data view to mdi widget
        sub = QMdiSubWindow()
        sub.setWidget(self.dataView)
        #sub.setWindowTitle(dset_name + ': ' + os.path.basename(dset_fname))
        sub.setWindowTitle(dset_name)
        self.mdi.addSubWindow(sub)        
        sub.show()
        self.mdi.tileSubWindows()

        ## Display status
        self.statusbar.showMessage('Displaying dataset')
        
        ##-------
        ## Populate commands that will be written in a notebook

        ## Add cmds 
        cmds = ['']
        cmds.append('# Show dataset')
        cmds.append(dset_name + '.head()')
        cmds.append('')
        self.cmds.add_cmd(cmds)
        ##-------
        

    def PopulateTable(self, data):

        ### FIXME : Data is truncated to single precision for the display
        ### Add an option in settings to let the user change this
        data = data.round(3)

        model = PandasModel(data)
        self.dataView = QtWidgets.QTableView()
        self.dataView.setModel(model)

    def PopulateSelect(self):

        #get data column header names
        colNames = self.data_model_arr.datasets[self.active_index].data.columns.tolist()

        #add the list items to comboBox
        self.ui.comboBoxSelect.blockSignals(True)
        self.ui.comboBoxSelect.clear()
        self.ui.comboBoxSelect.addItems(colNames)
        self.ui.comboBoxSelect.blockSignals(False)

    def OnSelColChanged(self):
        
        ## Threshold to show categorical values for selection
        selcol = self.ui.comboBoxSelCol.currentText()
        dftmp = self.data_model_arr.datasets[self.active_index].data[selcol]
        val_uniq = dftmp.unique()
        num_uniq = len(val_uniq)

        self.ui.comboSelVals.show()

        ## Select values if #unique values for the field is less than set threshold
        if num_uniq <= self.TH_NUM_UNIQ:
            #self.ui.wFilterNumerical.hide()
            #self.ui.wFilterCategorical.show()
            self.PopulateComboBox(self.ui.comboSelVals, val_uniq)
        
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

    
    def OnSelIndexChanged(self):
        
        sel_col = self.ui.comboSelVar.currentText()
        sel_colVals = self.data_model_arr.datasets[self.active_index].data[sel_col].unique()
        
        if len(sel_colVals) < self.TH_NUM_UNIQ:
            self.ui.comboSelVal.show()
            self.PopulateComboBox(self.ui.comboSelVal, sel_colVals)
        else:
            print('Too many unique values for selection, skip : ' + str(len(sel_colVals)))

    
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

        self.PopulateComboBox(self.ui.comboNormVar, colNames, '--var name--')

        #self.ui.comboOutVar.checkAllItems()

        #self.ui.comboSelVal.hide()


