# This Python file uses the following encoding: utf-8
"""
contact: software@cbica.upenn.edu
Copyright (c) 2018 University of Pennsylvania. All rights reserved.
Use of this source code is governed by license located in license file: https://github.com/CBICA/NiChart_Viewer/blob/main/LICENSE
"""
from PyQt5 import QtCore, QtWidgets
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib as mpl
mpl.use('QT5Agg')


class PlotCanvas(QtWidgets.QWidget):
    """ A generic Plotting class that derives from FigureCanvasQTAgg
    and plots data as per different options"""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        super(PlotCanvas,self).__init__()
        """The constructor."""

        # a figure instance to plot on
        self.fig = mpl.figure.Figure(figsize=(width, height), dpi=dpi)

        #FigureCanvas
        self.canvas = FigureCanvas(self.fig)
        self.setParent(parent)

        #toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.setLayout(QtWidgets.QVBoxLayout())

        #self.layout().addWidget(self.toolbar)
        self.layout().addWidget(self.canvas)

    def getFigure(self):
        return self.fig

    def draw(self):
        self.canvas.draw()
