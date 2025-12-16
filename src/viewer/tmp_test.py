import streamlit as st

# Initialize session state if the key doesn't exist
if 'my_range_values' not in st.session_state:
    st.session_state.my_range_values = (25, 75) # Default min and max values

st.title("Streamlit Range Slider with Session State")

# Create the range slider, linking its value to st.session_state.my_range_values
selected_range = st.slider(
    "Select a range of values",
    min_value=0,
    max_value=100,
    value=st.session_state.my_range_values, # Use the value from session state
    key='my_range_values' # Link the slider to this session state key
)

st.write(f"The selected range is: {selected_range}")
st.write(f"Values in session state: {st.session_state.my_range_values}")

# You can also explicitly update session state if needed (though the key handles it for sliders)
# st.session_state.my_range_values = selected_range    
