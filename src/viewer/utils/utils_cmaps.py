import streamlit as st
import matplotlib.colors as mcolors
import streamlit_antd_components as sac
import plotly.express as px

# Color maps for plots
cmaps_init = {
    'data': {
        'd1': [230, 0, 0],
        'd2': [0, 230, 0],
        'd3': [0, 0, 230],
        'd4': [230, 0, 0],
        'd5': [0, 230, 0],
        'd6': [0, 0, 230],
    },
    'centiles': {
        'centile_5': [200, 225, 255],
        'centile_25': [130, 180, 240],
        'centile_50': [0, 60, 130],
        'centile_75': [130, 180, 240],
        'centile_95': [200, 225, 255],
    },
    'fit lines': {
        'linfit': [221, 0, 0],
        'conf95': [153, 0, 0],
        'lowess': [51, 0, 0],
    }
}

alphas_init = {
    'data': 1.0,
    'centiles': 1.0,
    'fit lines': 1.0
}


def rgb_picker(rgb_init=[255, 0, 0], label="Pick a color"):
    '''
    Pick a color
    '''
    [r, g, b] = rgb_init
    hex_color = '#%02x%02x%02x' % (r, g, b)
    
    picked_color = st.color_picker(
        label,
        hex_color
    )
    
    r = int(picked_color[1:3], 16)
    g = int(picked_color[3:5], 16)
    b = int(picked_color[5:7], 16)
    
    rgba_new = [r, g, b]
    
    return rgba_new

def alpha_picker(alpha_init = 1.0, label="Pick alpha"):
    '''
    Pick a alpha value
    '''
    st.markdown(f'##### {label}')
    new_alpha = st.slider(
        "Transparency (alpha)",
        0.0,
        1.0,
        alpha_init,
        step=0.01,
        key = f'_slider_{label}',
        label_visibility="collapsed"
    )
    return new_alpha


def panel_update_cmaps():
    '''
    Update color maps and alpha for plots
    '''
    # Select color maps
    cmaps = st.session_state.plot_settings['cmaps']    
    cmaps_new = cmaps.copy()    
    num_max = max(len(inner_dict) for inner_dict in cmaps.values())

    st.markdown("##### Select colors:")
    for mcat in ['data', 'centiles']:
        with st.container(border=True):
            st.markdown(f'##### {mcat}')
            cols = st.columns(num_max)
            for i, (key,val) in enumerate(cmaps_new[mcat].items()):
                with cols[i]:
                    new_color = rgb_picker(val, key)
                    cmaps_new[mcat][key] = new_color

    # Select alpha values
    alphas = st.session_state.plot_settings['alphas']
    alphas_new = alphas.copy()

    st.markdown("##### Select alpha values:")
    with st.container(border=True):
        cols = st.columns(len(alphas_new))
        for i, (key, val) in enumerate(alphas_new.items()):
            with cols[i]:
                new_alpha = alpha_picker(val, key)
                alphas_new[key] = new_alpha

    # Update/reset values
    if st.button('Reset to default'):
        st.success('Color maps are reset to default values')
        st.session_state.plot_settings['cmaps'] = cmaps_init.copy()
        st.session_state.plot_settings['alphas'] = alphas_init.copy()
        st.rerun()

    if st.button('Select'):
        st.success('Color maps are updated to new values')
        st.session_state.plot_settings['cmaps'] = cmaps_new.copy()
        st.session_state.plot_settings['alphas'] = alphas_new.copy()


