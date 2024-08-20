# This Python file uses the following encoding: utf-8
"""
contact: software@cbica.upenn.edu
Copyright (c) 2018 University of Pennsylvania. All rights reserved.
Use of this source code is governed by license located in license file: https://github.com/CBICA/NiChart_Viewer/blob/main/LICENSE
"""

from PyQt5 import QtCore, QtGui, QtWidgets, uic
from yapsy.PluginManager import PluginManager
from yapsy.IPlugin import IPlugin
import os, sys
import numpy as np
from NiChart_Viewer.core.dataio import DataIO
from NiChart_Viewer.core.model.datamodel import DataModel, DataModelArr
from NiChart_Viewer.core.model.cmdmodel import CmdModel
from .aboutdialog import AboutDialog
#from NiChart_Viewer.resources import resources
from PyQt5.QtWidgets import QAction
import pandas as pd
from NiChart_Viewer.core.baseplugin import BasePlugin
from NiChart_Viewer.core import iStagingLogger

logger = iStagingLogger.get_logger(__name__)


class MainWindow(QtWidgets.QMainWindow):
    
    def __init__(self, dataFiles = None, dictFiles = None):
        super(MainWindow,self).__init__()

        logger.info('New NiChart_Viewer session starting...')

        self.SetupUi()
        self.SetupConnections()

        # Variable to keep all datasets
        self.data_model_arr = DataModelArr()

        # Variable to keep all commands (to create a notebook)
        self.cmds = CmdModel()
        
        # Widget for err messages
        self.errmsg = QtWidgets.QErrorMessage()
        
        # Variable to keep path for the last data file loaded
        self.dataPathLast = ''

        ## FIXME : Tmp
        root = os.path.dirname(__file__)
        self.root_dir = os.path.dirname(__file__)

        self.dataPathLast = root

        # Create plugin manager
        self.manager = PluginManager(categories_filter={ "Tabs": BasePlugin})
        self.manager.setPluginPlaces([os.path.join(root, 'plugins')])

        # Load plugins
        self.manager.locatePlugins()
        self.manager.loadPlugins()

        ## Create a list of plugin objects, names and tab position
        plTmp = []
        plNameTmp = []
        plIndTmp = []
        pluginDescriptions = []
        for plugin in self.manager.getAllPlugins():
            
            po = plugin.plugin_object
            po.data_model_arr = self.data_model_arr
            po.root_dir = self.root_dir
            po.cmds = self.cmds
            po.statusbar = self.ui.statusbar
            po.errmsg = self.errmsg
            po.SetupConnections()
    
            plTmp.append(po)
            plNameTmp.append(plugin.name)
            plIndTmp.append(po.getTabPosition())
            pluginDescriptions.append(plugin.description)
            
        ## Sort plugins based on tab position. Create a dictionary of plugins
        indSort = np.argsort(plIndTmp)
        
        plTmp = np.array(plTmp)[indSort]
        plNameTmp = np.array(plNameTmp)[indSort]
        

        self.pluginDescriptions = np.array(pluginDescriptions)[indSort]       # A list with plugin descriptions
        
        self.Plugins = dict(zip(plNameTmp, plTmp))                  # A dictionary with plugin name and plugin object
        
        #logger.info(self.Plugins)
        #logger.info('AAAAAA')
        
        self.IndexPlugins = dict(zip(plNameTmp, np.arange(0, plNameTmp.shape[0])))  # A dictionary with plugin name and plugin index
        

        logger.info("Loaded Plugins: %s", self.Plugins.keys())
        logger.info(self.IndexPlugins)

        # Create new menu action for each plugin
        for i, key in enumerate(self.Plugins.keys()):
            newAction = QAction(key, self)        
            newAction.setCheckable(True)
            #newAction.setShortcut('Ctrl+N')
            #newAction.setStatusTip('New document')

            newAction.setChecked(True)      ## Initially tabs are visible
            
            ## The "function factory" call is an alternative to using a lambda function
            ##   The makeXXX function factory takes an arg and returns a function XXX
            newAction.triggered.connect(self.makeOnPluginChecked(key))

            self.ui.menuPlugins.addAction(newAction)

        ## Add a tab for each plugin
        for i, [key,value] in enumerate(self.Plugins.items()):
            self.ui.tabWidget.insertTab(i, value, key)
            if i>0:
                self.ui.tabWidget.setTabVisible(i, False)
                self.ui.tabWidget.setTabVisible(i, True)
                
        if dataFiles is not None:
            # if datafile provided on cmd line, load it
            #self.Plugins['Table View'].LoadDataFile(dataFile)
            for dTmp in dataFiles:
                self.LoadDataFile(dTmp)
        
        if dictFiles is not None:
            for dTmp in dictFiles:
                self.LoadDictFile(dTmp)

        ## Info panel        
        self.ui.wInfo.setStyleSheet('background-color : rgb(10, 10, 50); color: rgb(230, 230, 230)')

        ## Statusbar
        self.ui.statusbar.setStyleSheet('background-color : rgb(100, 100, 170); color: rgb(230, 230, 210)')
        self.ui.statusbar.setFont(QtGui.QFont('Times', 14))

        #self.ui.wInfo.setMinimumSize(600, 200)
        #self.SetHelpMsg()

        self.ui.wInfo.hide()

        #self.actionSaveData.setVisible(False)

        #self.actionSaveData.setEnabled(False)
        #self.actionSaveNotebook.setEnabled(False)
        
        self.actionHelpConsole.setCheckable(True)
        
        ## Include Mac menu bar
        #self.ui.menuFile.setMenuRole(QAction.NoRole)
        #self.ui.actionOpen.setMenuRole(QAction.NoRole)
        #self.ui.actionSave.setMenuRole(QAction.NoRole)

    def __del__(self):
        logger.info('NiChart_Viewer session ending...')

    def SetupConnections(self):
        self.actionOpenData.triggered.connect(self.OnLoadDsetClicked)
        self.actionSaveData.triggered.connect(self.OnSaveDataClicked)
        #self.actionSaveNotebook.triggered.connect(self.OnSaveNotebookClicked)
        self.actionAbout.triggered.connect(self.OnAboutClicked)
        self.actionHelpConsole.triggered.connect(self.OnHelpConsoleChecked)
        self.ui.tabWidget.currentChanged.connect(self.OnTabChanged)


    #def makeOnPluginChecked(self, checked):
        #logger.info(checked)

    def makeOnPluginChecked(self, pluginName):
        def OnPluginChecked(checked):
            tabIndex = self.IndexPlugins[pluginName]
            if checked==True:
                self.ui.tabWidget.setTabVisible(tabIndex, True)
                self.ui.statusbar.showMessage('Plugin activated: ' + pluginName, 8000)
                
            else:
                self.ui.tabWidget.setTabVisible(tabIndex, False)
                self.ui.statusbar.showMessage('Plugin deactivated: ' + pluginName, 8000)
        

        return OnPluginChecked


    def OnTabChanged(self):
        
        indTab = self.ui.tabWidget.currentIndex()
        txtHelp = self.pluginDescriptions[indTab]

        txtPlugin = list(self.Plugins.keys())[indTab]

        self.ui.wInfo.setText(txtHelp)
        self.ui.statusbar.showMessage('Selected plugin: ' + txtPlugin, 8000)
 
    def OnHelpConsoleChecked(self):
        if self.actionHelpConsole.isChecked():
            self.ui.wInfo.show()
            self.ui.statusbar.showMessage('Help console activated', 8000)

        else:
            self.ui.wInfo.hide()
            self.ui.statusbar.showMessage('Help console deactivated', 8000)
        
 
    def SetupUi(self):
        root = os.path.dirname(__file__)
        self.ui = uic.loadUi(os.path.join(root, 'mainwindow.ui'), self)
        self.ui.setWindowTitle('NiChart_Viewer')
        #self.setWindowIcon(QtGui.QIcon(":/images/NiChart_ViewerLogo.png"))
        self.setWindowIcon(QtGui.QIcon(os.path.join(root, 'resources', 'NiChart_ViewerLogo.png')))
        self.aboutdialog = AboutDialog(self)

    def LoadDataFile(self, filename):
    
        root = os.path.dirname(__file__)

        # Read data file
        dio = DataIO()
        if filename.endswith('.pkl.gz') | filename[0].endswith('.pkl'):
            d = dio.ReadPickleFile(filename)
        elif filename.endswith('.csv'):
            d = dio.ReadCSVFile(filename)
        else:
            d = None

        ## FIXME: This is fixed to a single hard-coded MUSE dictionary 
        ##        and a smaller set of ROIs for now
        ##        It will be a separate generic action in the GUI in next versions
        if (d is not None):

            # Apply dictionary for column names
            
            ## Read dictionary
            dict_csv = os.path.join(root, 'shared', 'dictionaries', 'dicts_dlmuse',
                         'MUSE_ROI_Names_NiChart_Selection_V2.2.csv')
            
            fMuseDict = os.path.join(dict_csv)
            
            dio = DataIO()
            df_musedict = dio.ReadCSVFile(fMuseDict)
            
            keys = df_musedict.MUSE_ROI_Index.astype('str').tolist()
            values = df_musedict.MUSE_ROI_Name.astype('str').tolist()
            musedict = dict(zip(keys, values))
            
            all_keys = all([x in d.columns for x in keys])
            
            if all_keys == True: 
                d = d[d.columns[0:1].tolist() + keys]
            d = d.rename(columns = musedict)


        # Load data to model
        if (d is not None):

            logger.info('New data read from file: %s', filename)
            dmodel= DataModel(d, filename)
            self.data_model_arr.AddDataset(dmodel)

            self.actionSaveData.setEnabled(True)
            #self.actionSaveNotebook.setEnabled(True)

            ## Call signal for change in data
            self.data_model_arr.OnDataChanged()

            self.ui.statusbar.showMessage('Loaded dataset: ' + filename, 8000)

        else:
            logger.warning('Loaded data was not valid.')
            self.ui.statusbar.showMessage('WARNING: Could not load dataset: ' + filename)


        ##-------
        ## Populate commands that will be written in a notebook
        if (d is not None):
            dset_name = self.data_model_arr.dataset_names[self.data_model_arr.active_index]            
            cmds = ['']
            cmds.append('# Load dataset')
            cmds.append(dset_name + ' = pd.read_csv("' + filename + '")')
            self.cmds.add_cmd(cmds)
        ##-------

    def OnLoadDsetClicked(self):
        
        if self.dataPathLast == '':
            directory = QtCore.QDir().homePath()
        else:
            directory = self.dataPathLast
        
        filename = QtWidgets.QFileDialog.getOpenFileName(None,
            caption = 'Open data file',
            directory = directory,
            filter = "Pickle/CSV files (*.pkl.gz *.pkl *.csv)")

        if filename[0] == "":
            logger.warning("No file was selected")
        else:
            self.LoadDataFile(filename[0])
            self.dataPathLast = os.path.dirname(filename[0])

    ## Function to write commands into a notebook
    ##   Commands are collected from individual actions within plugins
    def OnSaveNotebookClicked(self):

        if self.dataPathLast == '':
            directory = QtCore.QDir().homePath()
        else:
            directory = self.dataPathLast
        
        filename = QtWidgets.QFileDialog.getSaveFileName(None,
            caption = 'Save as a notebook',
            directory = directory,
            filter = "Jupyter notebook files (*.ipynb)")[0]
        if filename[-6:] != '.ipynb':
            filename = filename + '.ipynb'

        self.cmds.add_cmd([''])          # Add a single empty line to indicate end of block
        
        logger.info('--------------------------------------------')
        logger.info(self.cmds.cmds)
        logger.info('--------------------------------------------')
        
        self.cmds.cmds_to_notebook(filename)
        
        self.ui.statusbar.showMessage('Notebook saved to: ' + filename, 8000)


    ## Function to write current data frame to csv file
    def OnSaveDataClicked(self):
        filename = QtWidgets.QFileDialog.getSaveFileName(None,
            caption = 'Save as a csv file',
            directory = QtCore.QDir().homePath(),
            filter = "CSV files (*.csv)")[0]
        if filename[-4:] != '.csv':
            filename = filename + '.csv'
        
        self.data_model_arr.datasets[self.data_model_arr.active_index].data.to_csv(filename, index=False)

        self.ui.statusbar.showMessage('Datafile saved to: ' + filename, 8000)

        ##-------
        ## Populate commands that will be written in a notebook
        dset_name = self.data_model_arr.dataset_names[self.data_model_arr.active_index]        
        cmds = ['']
        cmds.append('Saving data file')        
        cmds.append('pd.to_csv( "' + dset_name + '", index=False)')
        cmds.append('')
        self.cmds.add_cmd(cmds)
        ##-------

    def OnAboutClicked(self):
        self.aboutdialog.show()
        self.ui.statusbar.showMessage('Information console activated', 8000)
        

    def OnCloseClicked(self):
        #close currently loaded data and model
        QtWidgets.QApplication.quit()

    def ResetUI(self):
        #reset all UI
        pass
