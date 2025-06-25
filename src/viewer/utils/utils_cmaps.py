import streamlit as st
import matplotlib.colors as mcolors

def color_picker_with_alpha(
    init_color="#ff0000",
    init_alpha=1
):
    '''
    Pick a color and alpha value
    '''
    hex_color = st.color_picker(
        "Pick a color", init_color
    )
    alpha = st.slider(
        "Transparency (alpha)", 0.0, 1.0, init_alpha, step=0.01
    )
    
    rgb = mcolors.to_rgb(hex_color)  # (r, g, b) in [0,1]
    
    rgba = f"rgba({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)}, {alpha})"
    
    return rgba

def panel_update_cmap(cmap):
    '''
    Update color map for plots
    '''
    st.markdown("##### Select a color for each plot element")

    # Create a section with color pickers
    cmap_out = cmap.copy()
    
    cols = st.columns(len(cmap))
    for i,item in enumerate(cmap.items()):
        with cols[i]:
            cmap_out[item[0]] = st.color_picker(f"{item[0]}", value=item[1])

    bcols = st.columns([1,1,10])
    with bcols[0]:
        if st.button('Reset'):
            st.success('Reset to init vals')
            
    with bcols[1]:
        if st.button('Select'):
            st.success('Updated vals')
            return cmap_out
        
    return cmap
