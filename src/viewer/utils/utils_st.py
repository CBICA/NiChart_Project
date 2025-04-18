import os
import shutil
from typing import Any

import numpy as np
import streamlit as st
import utils.utils_io as utilio
import utils.utils_session as utilses

def get_next_option(list_options, sel_opt):
    '''
    From a list of options get the next one to selected option
    '''
    sel_ind = list_options.index(sel_opt)
    new_ind = (sel_ind + 1) % len(list_options)
    next_opt = list_options[new_ind]
    return next_opt
