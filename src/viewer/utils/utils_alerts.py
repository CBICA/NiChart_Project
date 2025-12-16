import streamlit as st

def render_alert():
    if "alert_message" in st.session_state:
        alert_type_funcs = {
            'info': st.info,
            'success': st.success,
            'warning': st.warning,
            'error': st.error
        }
        if "alert_type" not in st.session_state:
            alert_type = "info"
        alert_type = st.session_state.alert_type
        alert_message = st.session_state.alert_message
        # Invoke the alert function with the given message
        alert_type_funcs[alert_type](alert_message)
        # Delete the alert
        del st.session_state.alert_message
        if "alert_type" in st.session_state:
            del st.session_state.alert_type

def alert(msg, type='info'):
    '''
    This function puts an alert into the session state for rendering at next page load.
    '''
    st.session_state.alert_message = msg
    st.session_state.alert_type = type