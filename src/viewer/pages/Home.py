import streamlit as st
import utils.utils_menu as utilmenu
import utils.utils_session as utilss
import utils.utils_st as utilst
from streamlit_extras.stylable_container import stylable_container
from streamlitextras.webutils import stxs_javascript
import utils.utils_pages as utilpg

# Page config should be called for each page
utilss.config_page()

utilpg.select_main_menu('Home')

with st.container(border=True):
    st.markdown(
        """
        ### Welcome to NiChart Project!
        
        - NiChart is an open-source framework built specifically for deriving Machine Learning based indices from MRI data.
               
        - You can install NiChart Project desktop
        ```
        pip install NiChart_Project
        ```

        - Run the application
        ```
        cd src/viewer
        streamlit run NiChartProject.py
        ```

        - Alternatively, the cloud app can be launched at
        ```
        https://cloud.neuroimagingchart.com
        ```

        - Check out [NiChart Web page](https://neuroimagingchart.com)
        - Visit [NiChart GitHub](https://github.com/CBICA/NiChart_Project)
        - Jump into [our documentation](https://cbica.github.io/NiChart_Project)
        - Ask a question in [our community discussions](https://github.com/CBICA/NiChart_Project/discussions)
        """
    )
