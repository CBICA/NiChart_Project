#import streamlit as st
#import pandas as pd

#with st.container(border=True):

    ## Get df columns
    #df = pd.read_csv('/home/gurayerus/Desktop/DDD/rois.csv')

    #vType = df.Type.unique()

    #xvar = st.selectbox("X Var", vType, key="plot_x1", index=None)

    #if xvar is None:
        #vType2 = []
    #else:
        #vType2 = df[df.Type == xvar].Name.tolist()
    #xvar2 = st.selectbox("X2 Var", vType2, key="plot_x2", index=None)


#import streamlit as st
#if 'checkbox_label' not in st.session_state:
    #st.session_state.checkbox_label = "Original Label"
#checkbox_value = st.checkbox(st.session_state.checkbox_label)
#st.session_state.checkbox_label = "Updated Label"
#with st.container():
    #st.write(st.session_state)

import streamlit as st
if "instantiated" not in st.session_state:
    st.session_state.wtext= 'HELLO'
    st.session_state.instantiated = True
show_panel_experiment = st.checkbox(st.session_state.wtext)
if show_panel_experiment:
    with st.container(border=True):
        if st.checkbox('Select'):
            st.session_state.wtext = 'BYE'
        st.write('Success!')
with st.container():
    st.write(st.session_state)


#with st.popover("Open popover"):
    #st.markdown("Hello World 👋")
    #name = st.text_input("What's your name?")

#st.write("Your name:", name)

#def display_folder_contents(folder_path: str, parent_folder: str = "") -> None:
    #"""Displays the contents of a folder in a Streamlit panel with a tree structure.

    #Args:
        #folder_path (str): The path to the folder.
        #parent_folder (str): The parent folder's name (optional).
    #"""

    #st.title("Folder Contents")

    ## Check if the folder exists
    #if not os.path.exists(folder_path):
        #st.error(f"Folder '{folder_path}' does not exist.")
        #return

    ## Get a list of files and directories in the folder
    #contents = os.listdir(folder_path)

    ## Create a container for the folder contents
    #container = st.container()

    ## Display the parent folder name
    #if parent_folder:
        #container.markdown(f"**{parent_folder}**")

    ## Iterate over the contents and display them
    #for item in contents:
        #item_path = os.path.join(folder_path, item)

        ## Check if the item is a file or a directory
        #if os.path.isfile(item_path):
            ## Display the file name with indentation based on the parent folder
            #file_name = os.path.basename(item_path)
            #file_url = f"download/{file_name}"  # Adjust the download URL as needed
            #container.markdown(
                #f"{'  ' * len(parent_folder.split('/'))}[Download]({file_url}) {file_name}"
            #)
        #else:
            ## Display the directory name with indentation and a link to explore it
            #directory_name = os.path.basename(item_path)
            #container.markdown(
                #f"{'  ' * len(parent_folder.split('/'))}[Explore]({directory_name}) {directory_name}"
            #)

            ## Recursively display the contents of the subdirectory
            #display_folder_contents(item_path, parent_folder=directory_name)

