# This Python file uses the following encoding: utf-8
"""
contact: software@cbica.upenn.edu
Copyright (c) 2018 University of Pennsylvania. All rights reserved.
Use of this source code is governed by license located in license file: https://github.com/CBICA/NiChart_Viewer/blob/main/LICENSE
"""

from yapsy.IPlugin import IPlugin
import configparser, os, sys

class BasePlugin(IPlugin):

        def __init__(self):
                """
                init
                """
                # initialise parent class
                IPlugin.__init__(self)


        def activate(self):
                """
                On activation tell that this has been successfull.
                """
                # get the automatic procedure from IPlugin
                IPlugin.activate(self)
                return


        def deactivate(self):
                """
                On deactivation check that the 'activated' flag was on then
                tell everything's ok to the test procedure.
                """
                IPlugin.deactivate(self)

        def readAdditionalInformation(self,filename):
            #TODO: extend this function to read more information
            basename = os.path.basename(filename)
            suffix = '.yapsy-plugin'
            configfile = os.path.join(filename,basename + suffix)
            cfg = configparser.ConfigParser()
            cfg.read(configfile)
            if(cfg.has_section("Tab")):
                if(cfg.has_option("Tab","Position")):
                    self.tabPosition = cfg.getint("Tab","Position")

        def getTabPosition(self):
            return self.tabPosition
