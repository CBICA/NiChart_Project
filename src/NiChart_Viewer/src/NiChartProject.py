import streamlit as st
from PIL import Image

from st_pages import add_page_title, get_nav_from_toml
from st_pages import hide_pages

nicon = Image.open("../resources/nichart1.png")
st.set_page_config(page_title="NiChart", page_icon=nicon, layout='wide',
                   menu_items = {'Get help':'https://neuroimagingchart.com/',
                                 'Report a bug':'https://neuroimagingchart.com/',
                                 'About':'https://neuroimagingchart.com/'}
                  )

# If you want to use the no-sections version, this
# defaults to looking in .streamlit/pages.toml, so you can
# just call `get_nav_from_toml()`
nav = get_nav_from_toml(".streamlit/pages_sections.toml")

pg = st.navigation(nav)
add_page_title(pg)
pg.run()
