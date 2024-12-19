Developers - API
===========================

Submodules
----------
            df_muse = pd.DataFrame(
                columns=['MRID', '702', '701', '600', '601', '...'],
                data=[
                    ['Subj1', '...', '...', '...', '...', '...'],
                    ['Subj2', '...', '...', '...', '...', '...'],
                    ['Subj3', '...', '...', '...', '...', '...'],
                    ['...', '...', '...', '...', '...', '...']
                ]
            )
            st.markdown(
                """
                ### DLMUSE File:
                The DLMUSE CSV file contains volumes of ROIs (Regions of Interest) segmented by the DLMUSE algorithm. This file is generated as output when DLMUSE is applied to a set of images.
                """
            )
            st.write('Example MUSE data file:')
            st.dataframe(df_muse)
