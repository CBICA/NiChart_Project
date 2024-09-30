import plotly.express as px
import os
import streamlit as st
import tkinter as tk
from tkinter import filedialog




def display_folder_contents(folder_path, parent_folder=""):
    """Displays the contents of a folder in a Streamlit panel with a tree structure.

    Args:
        folder_path (str): The path to the folder.
        parent_folder (str): The parent folder's name (optional).
    """

    st.title("Folder Contents")

    # Check if the folder exists
    if not os.path.exists(folder_path):
        st.error(f"Folder '{folder_path}' does not exist.")
        return

    # Get a list of files and directories in the folder
    contents = os.listdir(folder_path)

    # Create a container for the folder contents
    container = st.container()

    # Display the parent folder name
    if parent_folder:
        container.markdown(f"**{parent_folder}**")

    # Iterate over the contents and display them
    for item in contents:
        item_path = os.path.join(folder_path, item)

        # Check if the item is a file or a directory
        if os.path.isfile(item_path):
            # Display the file name with indentation based on the parent folder
            file_name = os.path.basename(item_path)
            file_url = f"download/{file_name}"  # Adjust the download URL as needed
            container.markdown(f"{'  ' * len(parent_folder.split('/'))}[Download]({file_url}) {file_name}")
        else:
            # Display the directory name with indentation and a link to explore it
            directory_name = os.path.basename(item_path)
            container.markdown(f"{'  ' * len(parent_folder.split('/'))}[Explore]({directory_name}) {directory_name}")

            # Recursively display the contents of the subdirectory
            display_folder_contents(item_path, parent_folder=directory_name)


def display_folder(in_dir):
    '''
    Displays the contents of a folder in a Streamlit panel.
    '''

    st.title("Folder Contents")

    # Check if the folder exists
    if not os.path.exists(in_dir):
        st.error(f"Folder '{in_dir}' does not exist.")
        return

    # Get a list of files and directories in the folder
    contents = os.listdir(in_dir)

    # Create a container for the folder contents
    container = st.container()

    # Iterate over the contents and display them
    for item in contents:
        item_path = os.path.join(in_dir, item)

        # Check if the item is a file or a directory
        if os.path.isfile(item_path):
            # Display the file name with a link to download it
            file_name = os.path.basename(item_path)
            file_url = f"download/{file_name}"  # Adjust the download URL as needed
            container.write(f"[Download]({file_url}) {file_name}")
        else:
            # Display the directory name with a link to explore it
            directory_name = os.path.basename(item_path)
            container.write(f"[Explore]({directory_name}) {directory_name}")



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
        path_curr = st.text_input(label_txt, value = path_curr, help = help_msg)
    return path_curr

def user_input_select(label, selections, key, helpmsg):
    '''
    St selection box to selet a text from the user
    '''
    tmpcol = st.columns((1,8))
    with tmpcol[0]:
        user_sel = st.selectbox(label, selections, key = key, help = helpmsg)
    return user_sel

def show_img3D(img, scroll_axis, sel_axis_bounds, img_name):
    '''
    Displays a 3D img
    '''

    # Create a slider to select the slice index
    slice_index = st.slider(f"{img_name}",
                            0,
                            sel_axis_bounds[1] - 1,
                            value=sel_axis_bounds[2],
                            key = f'slider_{img_name}')

    # Extract the slice and display it
    if scroll_axis == 0:
        st.image(img[slice_index, :, :], use_column_width = True)
    elif scroll_axis == 1:
        st.image(img[:, slice_index, :], use_column_width = True)
    else:
        st.image(img[:, :, slice_index], use_column_width = True)


