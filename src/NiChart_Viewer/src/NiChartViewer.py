import streamlit as st
from PIL import Image

nicon = Image.open("../resources/nichart1.png")
st.set_page_config(
        page_title="NiChart Viewer",
        page_icon=nicon,
        )

st.sidebar.image("../resources/nichart1.png")
st.write("# Welcome to NiChart! 👋")

st.sidebar.info("""
                    Note: This website is based on materials from the [NiChart Project](https://neuroimagingchart.com/).
                    The content and the logo of NiChart are intellectual property of [CBICA](https://www.med.upenn.edu/cbica/).
                    Make sure that you read the [licence](https://github.com/CBICA/NiChart_Project/blob/main/LICENSE).
                    """)

with st.sidebar.expander("Acknowledgments"):
    st.markdown("""
                The CBICA Dev team
                """)



st.sidebar.success("Select a task above")

st.markdown(
        """
    NiChart is an open-source framework built specifically for
    deriving Machine Learning based indices from MRI data.

    **👈 Select a task from the sidebar** to view your derived
    data!

    ### Want to learn more?
    - Check out [NiChart Web page](https://neuroimagingchart.com)
    - Visit [NiChart GitHub](https://github.com/CBICA/NiChart_Project)
    - Jump into our [documentation](https://github.com/CBICA/NiChart_Project)
    - Ask a question in our [community
        forums](https://github.com/CBICA/NiChart_Project)
        """
)

st.markdown(
            """
            You can try NiChart manually via our github
            ```bash
            git clone https://github.com/CBICA/NiChart_Project
            git submodule update --init --recursive --remote
            pip install -r requirements.txt
            ```

            And to run the workflows, just run:
            ```bash
            python3 run.py --dir_input input folder --dir_output output_folder --studies 1 --version my_version --cores 4 --conda 0
            ```

            You can always find more options at our documentation
            """
)
