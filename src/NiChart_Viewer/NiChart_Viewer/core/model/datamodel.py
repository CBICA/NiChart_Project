# This Python file uses the following encoding: utf-8
"""
contact: software@cbica.upenn.edu
Copyright (c) 2018 University of Pennsylvania. All rights reserved.
Use of this source code is governed by license located in license file: https://github.com/CBICA/NiChart_Viewer/blob/main/LICENSE
"""

import pandas as pd
import numpy as np
import importlib.resources as pkg_resources
import os, sys
import joblib
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5 import QtCore
from NiChart_Viewer.core import iStagingLogger

logger = iStagingLogger.get_logger(__name__)

class PandasModel(QtCore.QAbstractTableModel):
    def __init__(self, data, parent=None):
        QtCore.QAbstractTableModel.__init__(self, parent)
        self._data = data
        self.header_labels = None

    def rowCount(self, parent=None):
        return len(self._data.values)

    def columnCount(self, parent=None):
        return self._data.columns.size

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        self.header_labels = self._data.keys()
        if role == QtCore.Qt.DisplayRole and orientation == QtCore.Qt.Horizontal:
            return self.header_labels[section]
        return QtCore.QAbstractTableModel.headerData(self, section, orientation, role)

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if index.isValid():
            if role == QtCore.Qt.DisplayRole:
                return QtCore.QVariant(str(
                    self._data.iloc[index.row()][index.column()]))
        return QtCore.QVariant()


class DataModelArr(QObject):
    """This class holds a collection of data models."""

    active_dset_changed = QtCore.pyqtSignal()

    def __init__(self):

        logger.info('In DataModelArr constructor')

        QObject.__init__(self)
        """The constructor."""
        
        ## Table view is very slow if there are many rows
        ## We limit it here as a shortcut solution
        ## FIXME: Find a good way of managing this
        self.TABLE_MAXROWS = 50
        
        self._datasets = []         ## An array of datasets
        self._dataset_names = []    ## Names of datasets (auto generated for now)
        self._active_index = -1     ## An index that keeps the index of active dataset

        logger.info('Exit DataModelArr constructor')

    ## Setter and Getter functions for all variables 
    ## https://stackoverflow.com/questions/2627002/whats-the-pythonic-way-to-use-getters-and-setters

    #############################
    ## decorators for datasets
    @property
    def datasets(self):
        return self._datasets

    @datasets.setter
    def datasets(self, value):
        self._datasets = value

    @datasets.deleter
    def datasets(self):
        del self._datasets

    #############################
    ## decorators for dataset_names
    @property
    def dataset_names(self):
        return self._dataset_names

    @dataset_names.setter
    def dataset_names(self, value):
        self._dataset_names = value

    @dataset_names.deleter
    def dataset_names(self):
        del self._dataset_names

    #############################
    ## decorators for active_index
    @property
    def active_index(self):
        return self._active_index

    @active_index.setter
    def active_index(self, value):
        self._active_index = value

    @active_index.deleter
    def active_index(self):
        del self._active_index

    #############################
    ## Function to add new dataset
    def AddDataset(self, value):

        logger.info('In DataModelArr.AddDataSet()')

        ## Get new index for dataset
        self.active_index = len(self.datasets)
        
        ## Add dataset
        self.datasets.append(value)
        
        ## Add dataset name        
        #self.dataset_names.append('DSET_' + str(self.active_index + 1))
        self.dataset_names.append(os.path.basename(value.file_name).replace('.csv', '').replace('.pkl.gz', ''))
        
        logger.info('Exit DataModelArr.AddDataSet()')
        
    #############################
    ## Emit signal when data was changed
    def OnDataChanged(self):
        logger.info('Signal to emit: active_dset_changed')
        self.active_dset_changed.emit()

class DataModel(QObject):
    """This class holds the data model."""

    dset_changed = QtCore.pyqtSignal()

    def __init__(self, data=None, fname=None, data_index=None):
        QObject.__init__(self)

        self._data = None        ## The dataset (a pandas dataframe)
        self._file_name = None   ## The name of the data file

        if data is not None: 
            self.data = data
            self.file_name = fname
            self.data_index = data_index

    ## Setter and Getter functions for all variables 
    ## https://stackoverflow.com/questions/2627002/whats-the-pythonic-way-to-use-getters-and-setters

    #############################
    ## decorators for data
    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @data.deleter
    def data(self):
        del self._data

    #############################
    ## decorators for file_name
    @property
    def file_name(self):
        return self._file_name
    @file_name.setter
    def file_name(self, value):
        self._file_name = value
    @file_name.deleter
    def file_name(self):
        del self._file_name

    #############################
    ## Check if dataset is valid
    def IsValidData(self, data=None):
        """Checks if the data is valid or not."""
        if data is None:
            data = self.data
        
        if not isinstance(data, pd.DataFrame):
            return False
        else:
            return True

    #############################
    ## Get data type of columns
    def GetColumnDataTypes(self):
        """Returns all header names for all columns in the dataset."""
        d = self.data.dtypes
        return d

    #############################
    ## Reset the data set
    def Reset(self):
        #clear all contents of data/model and release memory etc.
        #TODO: this needs to be done correctly,
        #is there a better way to clear data?
        del self.data
        self.data = None

