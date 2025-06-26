import streamlit as st
import matplotlib.colors as mcolors
import streamlit_antd_components as sac
import plotly.express as px

# Color maps for plots
cmaps_init = {
    'data': {
        'd1': [230, 0, 0, 1.0],
        'd2': [0, 230, 0, 1.0],
        'd3': [0, 0, 230, 1.0],
        'd4': [230, 0, 0, 1.0],
        'd5': [0, 230, 0, 1.0],
        'd6': [0, 0, 230, 1.0],
    },
    'centiles': {
        'Centile 5': [0, 0, 17, 1.0],
        'Centile 25': [0, 0, 51, 1.0],
        'Centile 50': [0, 0, 102, 1.0],
        'Centile 75': [0, 0, 153, 1.0],
        'Centile 95': [0, 0, 221, 1.0],
    },
    'fit lines': {
        'linfit': [221, 0, 0, 1.0],
        'conf95': [153, 0, 0, 1.0],
        'lowess': [51, 0, 0, 1.0],
    }
}

def rgba_picker(rgba_init=[255, 0, 0, 1.0], label="Pick a color"):
    '''
    Pick a color and alpha value
    '''
    [r, g, b, a] = rgba_init
    hex_color = '#%02x%02x%02x' % (r, g, b)
    
    picked_color = st.color_picker(
        label,
        hex_color
    )
    
    alpha = st.slider(
        "Transparency (alpha)",
        0.0,
        1.0,
        a,
        step=0.01,
        key = f'_slider_{label}'
    )
    
    r = int(picked_color[1:3], 16)
    g = int(picked_color[3:5], 16)
    b = int(picked_color[5:7], 16)
    
    rgba_new = [r, g, b, alpha]
    
    return rgba_new

def panel_update_cmap():
    '''
    Update color maps for plots
    '''
    cmaps = st.session_state.plot_settings['cmaps']    
    cmaps_new = cmaps.copy()    
    num_max = max(len(inner_dict) for inner_dict in cmaps.values())

    st.markdown("##### Select a color for each plot element")
    for mcat in cmaps_new.keys():
        with st.container(border=True):
            st.markdown(f'{mcat}')
            cols = st.columns(num_max)
            for i, (key,val) in enumerate(cmaps_new[mcat].items()):
                with cols[i]:
                    new_color = rgba_picker(val, key)
                    cmaps_new[mcat][key] = new_color

    if st.button('Reset to default'):
        st.success('Color maps are reset to default values')
        st.session_state.plot_settings['cmaps'] = cmaps_init.copy()
        st.rerun()

    if st.button('Select'):
        st.success('Color maps are updated to new values')
        st.session_state.plot_settings['cmaps'] = cmaps_new.copy()

