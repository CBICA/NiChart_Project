import streamlit as st
import matplotlib.colors as mcolors
import streamlit_antd_components as sac

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

def panel_update_cmap(cmap, alpha):
    '''
    Update color map for plots
    '''
    st.markdown("##### Select a color for each plot element")

    # Create a section with color pickers
    cmap_out = cmap.copy()
    alpha_out = alpha.copy()

    for mcat in cmap.keys():
        with st.container(border=True):
            st.markdown(f'{mcat}')
            cols = st.columns(5)
            for i, (key,val) in enumerate(cmap[mcat].items()):
                st.write(f'{i}  {key} {val}')
                with cols[i]:
                    new_col = st.color_picker(
                        f"{key}",
                        value = val
                    )
                    new_alpha = st.slider(
                        f"{key}",
                        value = alpha[mcat][key],
                        min_value = 0,
                        max_value = 100
                    )
                cmap_out[mcat][key] = new_col
                alpha_out[mcat][key] = new_alpha

    if st.button('Reset'):
        st.success('Reset to init vals')

    if st.button('Select'):
        st.success('Updated vals')
        return cmap_out

    return cmap
