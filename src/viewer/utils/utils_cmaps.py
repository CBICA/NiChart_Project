import streamlit as st
import matplotlib.colors as mcolors
import streamlit_antd_components as sac
import plotly.express as px

# Color maps for plots
cmaps_init = {
    'data': {
        'd1': 'rgba(230, 0, 0, 1.0)',
        'd2': 'rgba(0, 230, 0, 1.0)',
        'd3': 'rgba(0, 0, 230, 1.0)',
        'd4': 'rgba(230, 0, 0, 1.0)',
        'd5': 'rgba(0, 230, 0, 1.0)',
        'd6': 'rgba(0, 0, 230, 1.0)',
    },
    'centiles': {
        'Centile 5': 'rgba(0, 0, 17, 1.0)',
        'Centile 25': 'rgba(0, 0, 51, 1.0)',
        'Centile 50': 'rgba(0, 0, 102, 1.0)',
        'Centile 75': 'rgba(0, 0, 153, 1.0)',
        'Centile 95': 'rgba(0, 0, 221, 1.0)',
    },
    'fit lines': {
        'linfit': 'rgba(221, 0, 0, 1.0)',
        'conf95': 'rgba(153, 0, 0, 1.0)',
        'lowess': 'rgba(51, 0, 0, 1.0)',
    }
}

def rgba_picker(rgba_init='rgba(255, 0, 0, 1.0)', label="Pick a color"):
    '''
    Pick a color and alpha value
    '''
    rgba_list = rgba_init.replace('rgba(', '').replace(')', '').split(',')
    [r, g, b, a] = [float(x) if '.' in x else int(x) for x in rgba_list]
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
    
    rgba_new = f'rgba({r}, {g}, {b}, {alpha})'
    
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

