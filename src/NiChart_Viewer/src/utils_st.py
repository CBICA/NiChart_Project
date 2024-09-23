import plotly.express as px
import os
import streamlit as st
import tkinter as tk
from tkinter import filedialog

def browse_file(path_init):
    '''
    File selector
    Returns the file name selected by the user and the parent folder
    '''
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    out_path = filedialog.askopenfilename(initialdir = path_init)
    path_out = os.path.dirname(out_path)
    root.destroy()
    return out_path, path_out

def browse_folder(path_init):
    '''
    Folder selector
    Returns the folder name selected by the user
    '''
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    out_path = filedialog.askdirectory(initialdir = path_init)
    root.destroy()
    return out_path

def user_input_text(label, init_val, help_msg):
    '''
    St text field to read a text input from the user
    '''
    tmpcol = st.columns((1,8))
    with tmpcol[0]:
        user_sel = st.text_input(label, value = init_val, help = help_msg)
        return user_sel
    
def user_input_file(label_btn, key_btn, label_txt, init_path_dir, init_path_curr, help_msg):
    '''
    St button + text field to read an input file path from the user
    '''
    path_curr = init_path_curr
    path_dir = init_path_dir
    tmpcol = st.columns((8,1))
    with tmpcol[1]:
        if st.button(label_btn, key = key_btn):
            path_curr, path_dir = browse_file(path_dir)
            
    with tmpcol[0]:
        tmp_sel = st.text_input(label_txt, value = path_curr, help = help_msg)
        if os.path.exists(tmp_sel):
            path_curr = tmp_sel
    return path_curr, path_dir

def user_input_folder(label_btn, key_btn, label_txt, init_path_dir, init_path_curr, help_msg):
    '''
    St button + text field to read an input directory path from the user
    '''
    path_curr = init_path_curr
    path_dir = init_path_dir
    tmpcol = st.columns((8,1))
    with tmpcol[1]:
        if st.button(label_btn, key = key_btn):
            if path_curr == '':
                path_curr = browse_folder(path_dir)
            else:
                path_curr = browse_folder(path_curr)
                
    with tmpcol[0]:
        tmp_sel = st.text_input(label_txt, value = path_curr, help = help_msg)
        if os.path.exists(tmp_sel):
            path_curr = tmp_sel
    return path_curr

def user_input_select(label, selections, key, helpmsg):
    '''
    St selection box to selet a text from the user
    '''
    tmpcol = st.columns((1,8))
    with tmpcol[0]:
        user_sel = st.selectbox(label, selections, key = key, help = helpmsg)
    return user_sel


