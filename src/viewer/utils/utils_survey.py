def is_survey_completed() -> bool:
    # First, check for session-local skip_survey
    if "skip_survey" in st.session_state:
        if st.session_state.skip_survey:
            return True
    # Look in the base output dir for the "survey submitted" file
    # (This occurs regardless of cloud or local)
    user_dir = st.session_state.paths['out_dir']
    indicator_filepath = os.path.join(user_dir, "survey.txt")
    if os.path.exists(indicator_filepath):
        return True # Survey has been submitted
    else:
        return False