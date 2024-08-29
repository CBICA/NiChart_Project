from PyQt5.QtGui import *
from PyQt5 import QtGui, QtCore, QtWidgets, uic
from PyQt5.QtWidgets import QMdiArea, QMdiSubWindow, QTextEdit, QComboBox
import sys, os
import pandas as pd
import numpy as np
from NiChart_Viewer.core.dataio import DataIO
# import dtale
from NiChart_Viewer.core.baseplugin import BasePlugin
from NiChart_Viewer.core import iStagingLogger
from NiChart_Viewer.core.gui.SearchableQComboBox import SearchableQComboBox
from NiChart_Viewer.core.gui.CheckableQComboBox import CheckableQComboBox
from NiChart_Viewer.core.model.datamodel import DataModel, DataModelArr, PandasModel

logger = iStagingLogger.get_logger(__name__)

from NiChart_Viewer.core.datautils import *

class MergeView(QtWidgets.QWidget,BasePlugin):

    def __init__(self):
        super(MergeView,self).__init__()
        
        self.data_model_arr = None
        self.active_index = -1
        self.dataset2_index = -1
        
        self.cmds = None

        root = os.path.dirname(__file__)
        self.readAdditionalInformation(root)
        self.ui = uic.loadUi(os.path.join(root, 'mergeview.ui'),self)
        
        self.mdi = self.findChild(QMdiArea, 'mdiArea')       
        self.mdi.setBackground(QtGui.QColor(245,245,245,255))
                
        ## Panel for action
        self.ui.comboAction = QComboBox(self.ui)
        self.ui.comboAction.setEditable(False)
        self.ui.vlAction.addWidget(self.ui.comboAction)
        #self.PopulateComboBox(self.ui.comboAction, ['Concat', 'Merge'], '--action--')        
        self.PopulateComboBox(self.ui.comboAction, ['Merge'], '--action--')        
                
        ### Panel for dset1 merge variables selection
        self.ui.comboBoxMergeVar1 = CheckableQComboBox(self.ui)
        self.ui.comboBoxMergeVar1.setEditable(False)
        self.ui.vlComboMerge1.addWidget(self.ui.comboBoxMergeVar1)

        ## Panel for dset2 selection
        self.ui.comboBoxDataset2 = QComboBox(self.ui)
        self.ui.hlMergeDset2.addWidget(self.ui.comboBoxDataset2)

        ### Panel for dset2 merge variables selection
        self.ui.comboBoxMergeVar2 = CheckableQComboBox(self.ui)
        self.ui.comboBoxMergeVar2.setEditable(False)
        self.ui.vlComboMerge2.addWidget(self.ui.comboBoxMergeVar2)

        ## Panel for dset2 selection for concat
        self.ui.comboBoxConcatDset2 = QComboBox(self.ui)
        self.ui.hlConcatDset2.addWidget(self.ui.comboBoxConcatDset2)

        ## Default value in merge view is to create a new dset (not to overwrite the active dset)
        self.ui.check_createnew.hide()        
        #self.ui.check_createnew.setCheckState(QtCore.Qt.Checked)
                
        self.ui.wOptions.setMaximumWidth(300)
        
        self.ui.edit_activeDset.setReadOnly(True)

        ## Panel are shown based on selected actions
        self.ui.wMerge.hide()
        self.ui.wConcat.hide()

    def SetupConnections(self):
        self.data_model_arr.active_dset_changed.connect(lambda: self.OnDataChanged())        

        self.ui.comboBoxDataset2.currentIndexChanged.connect(lambda: self.OnDataset2Changed())

        self.ui.comboBoxConcatDset2.currentIndexChanged.connect(lambda: self.OnConcatDset2Changed())

        self.ui.comboAction.currentIndexChanged.connect(self.OnActionChanged)

        self.ui.mergeBtn.clicked.connect(lambda: self.OnMergeDataBtnClicked())
        self.ui.concatBtn.clicked.connect(lambda: self.OnConcatBtnClicked())

    def OnActionChanged(self):
        
        logger.info('Action changed')

        self.ui.wMerge.hide()
        self.ui.wConcat.hide()

        self.selAction = self.ui.comboAction.currentText()

        if self.selAction == 'Merge':
            self.ui.wMerge.show()
        
        if self.selAction == 'Concat':
            self.ui.wConcat.show()

        self.statusbar.showMessage('Action selected: ' + self.selAction, 8000)

    def OnConcatDset2Changed(self):
        logger.info('Dataset2 selection changed')
        selDsetName = self.ui.comboBoxConcatDset2.currentText()
        self.dataset2_index = np.where(np.array(self.data_model_arr.dataset_names) == selDsetName)[0][0]
        
        logger.info('Dataset2 changed to : ' + selDsetName)
        logger.info('Dataset2 index changed to : ' + str(self.dataset2_index))

    def OnDataset2Changed(self):
        logger.info('Dataset2 selection changed')
        selDsetName = self.ui.comboBoxDataset2.currentText()
        self.dataset2_index = np.where(np.array(self.data_model_arr.dataset_names) == selDsetName)[0][0]
        
        colNames = self.data_model_arr.datasets[self.dataset2_index].data.columns.tolist()
        
        self.PopulateComboBox(self.ui.comboBoxMergeVar2, colNames, '--var name--')
        self.ui.comboBoxMergeVar2.show()
        self.ui.label_mergeon2.show()
        
        logger.info('Dataset2 changed to : ' + selDsetName)
        logger.info('Dataset2 index changed to : ' + str(self.dataset2_index))

    def OnMergeDataBtnClicked(self):

        ## Read merge options
        dset_name = self.data_model_arr.dataset_names[self.active_index]        
        dset_name2 = self.data_model_arr.dataset_names[self.dataset2_index]        
        
        mergeOn1 = self.ui.comboBoxMergeVar1.listCheckedItems()
        mergeOn2 = self.ui.comboBoxMergeVar2.listCheckedItems()

        dfCurr = self.data_model_arr.datasets[self.active_index].data
        dfDset2 = self.data_model_arr.datasets[self.dataset2_index].data
        
        ## Calculate results
        
        res_tmp = DataMerge(dfCurr, dfDset2, mergeOn1, mergeOn2)
        if res_tmp['out_code'] != 0:
            self.errmsg.showMessage(res_tmp['out_msg'])
            return;
        df_out = res_tmp['df_out']

        ## Create new dataset or update current active dataset
        if self.ui.check_createnew.isChecked():
            dmodel = DataModel(df_out, dset_name + '+' + dset_name2)
            self.data_model_arr.AddDataset(dmodel)
            self.data_model_arr.OnDataChanged()

        else:
            self.data_model_arr.datasets[self.active_index].data = df_out

        ## Display the table
        self.statusbar.showMessage('Dataframe updated, size: ' + str(df_out.shape), 8000)        
        self.ShowTable()

        ## Call signal for change in data
        self.data_model_arr.OnDataChanged()
        
        ##-------
        ## Populate commands that will be written in a notebook
        str_mergeOn1 = ','.join('"{0}"'.format(x) for x in mergeOn1)
        str_mergeOn2 = ','.join('"{0}"'.format(x) for x in mergeOn2)

        cmds = ['']
        cmds.append('# Merge datasets')
        cmds.append(dset_name + ' = ' + dset_name + '.merge(' + dset_name2 + 
                    ', left_on = [' + str_mergeOn1 +'], right_on = [' + str_mergeOn2 + 
                    '], suffixes=["","_DUPLVARINDF2"])')
        
        cmds.append('## Note: Lines added to drop duplicate columns in dset2')
        cmds.append('colsKeep = ' + dset_name + '.columns[' + dset_name +
                    '.columns.str.contains("_DUPLVARINDF2")==False]')
        cmds.append(dset_name + ' = ' + dset_name + '[colsKeep]')
        cmds.append(dset_name + '.head()')
        cmds.append('')
        self.cmds.add_cmd(cmds)
                

    def OnConcatBtnClicked(self):

        ## Read merge options
        dset_name = self.data_model_arr.dataset_names[self.active_index]        
        dset_name2 = self.data_model_arr.dataset_names[self.dataset2_index]        
        
        dfCurr = self.data_model_arr.datasets[self.active_index].data
        dfDset2 = self.data_model_arr.datasets[self.dataset2_index].data
        
        ## Apply merge
        df_out = ConcatData(dfCurr, dfDset2)

        # Set updated dset
        self.data_model_arr.datasets[self.active_index].data = df_out

        ## Show table
        self.statusbar.showMessage('Dataframe updated, size: ' + str(df_out.shape), 8000)        
        self.ShowTable()

        ## Call signal for change in data
        self.data_model_arr.OnDataChanged()
        
        ##-------
        ## Populate commands that will be written in a notebook
        #cmds = ['']
        #cmds.append('# Merge datasets')
        #cmds.append(dset_name + ' = ' + dset_name + '.merge(' + dset_name2 + 
                    #', left_on = [' + str_mergeOn1 +'], right_on = [' + str_mergeOn2 + 
                    #'], suffixes=["","_DUPLVARINDF2"])')
        
        #cmds.append('## Note: Lines added to drop duplicate columns in dset2')
        #cmds.append('colsKeep = ' + dset_name + '.columns[' + dset_name +
                    #'.columns.str.contains("_DUPLVARINDF2")==False]')
        #cmds.append(dset_name + ' = ' + dset_name + '[colsKeep]')
        #cmds.append(dset_name + '.head()')
        #cmds.append('')
        #self.cmds.add_cmd(cmds)
        

    def PopulateTable(self, data):
        
        ### FIXME : Data is truncated to single precision for the display
        ### Add an option in settings to let the user change this
        data = data.round(3)
        
        model = PandasModel(data)
        self.dataView = QtWidgets.QTableView()
        self.dataView.setModel(model)

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
        
    def OnDataChanged(self):

        ## For the merge, there should be at least 2 datasets
        if len(self.data_model_arr.datasets) > 1:
     
            ## Make options panel visible
            self.ui.wOptions.show()
        
            ## Set fields for various options     
            self.active_index = self.data_model_arr.active_index
                
            ## Get data variables
            dataset = self.data_model_arr.datasets[self.active_index]
            dsetName = self.data_model_arr.dataset_names[self.active_index]
            colNames = dataset.data.columns.tolist()

            ## Set active dset name
            self.ui.edit_activeDset.setText(dsetName)

            ## Update Merge Vars for dset1 and dset2
            self.PopulateComboBox(self.ui.comboBoxMergeVar1, colNames, '--var name--')
            self.PopulateComboBox(self.ui.comboBoxMergeVar2, colNames, '--var name--')

            ## Var selection for dataset2 is hidden until the user selects the dataset
            self.ui.comboBoxMergeVar2.hide()
            self.ui.label_mergeon2.hide()
            
            dataset_names = self.data_model_arr.dataset_names.copy()
            dataset_names.remove(dsetName)
            
            self.PopulateComboBox(self.ui.comboBoxDataset2, dataset_names, '--dset name--')
            self.PopulateComboBox(self.ui.comboBoxConcatDset2, dataset_names, '--dset name--')

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

        ##-------
        ## Populate commands that will be written in a notebook

        ## Add cmds 
        cmds = ['']
        cmds.append('# Show dataset')
        cmds.append(dset_name + '.head()')
        cmds.append('')
        self.cmds.add_cmd(cmds)
        ##-------
