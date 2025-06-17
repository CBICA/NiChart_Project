# -*- coding: utf-8 -*-
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st

def verify_data_dlmuse():
    flag_data = True
    return flag_data

def verify_data(method):
    flag_data = False
    if method == 'dlmuse':
        flag_data = verify_data_dlmuse()
    
    return flag_data
        
    









