# This Python file uses the following encoding: utf-8
"""
contact: software@cbica.upenn.edu
Copyright (c) 2018 University of Pennsylvania. All rights reserved.
Use of this source code is governed by license located in license file: https://github.com/CBICA/NiChart_Viewer/blob/main/LICENSE
"""

import logging
import sys, os
from logging.handlers import TimedRotatingFileHandler

logging.basicConfig(level=logging.INFO)

#FORMATTER = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")
FORMATTER = logging.Formatter("[%(asctime)s — %(name)s — %(filename)s: Line:%(lineno)s — %(funcName)20s() ] — %(levelname)s — %(message)s")
LOG_FILE = os.path.expanduser(os.path.join('~', '.NiChart_Viewer','NiChart_Viewer.log'))


def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)


    return console_handler

def get_file_handler():
    if not os.path.exists(os.path.dirname(LOG_FILE)):
        os.makedirs(os.path.dirname(LOG_FILE))
    file_handler = TimedRotatingFileHandler(LOG_FILE, when='midnight')
    file_handler.setFormatter(FORMATTER)
    return file_handler

def get_logger(logger_name):
    logger = logging.getLogger(logger_name)
    #logger.setLevel(logging.DEBUG) # better to have too much log than not enough
    logger.setLevel(logging.WARNING)
    logger.addHandler(get_console_handler())
    logger.addHandler(get_file_handler())
    logger.propagate = False
    return logger
