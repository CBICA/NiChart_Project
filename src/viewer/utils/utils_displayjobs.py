import streamlit as st
import time

def display_jobs():
    if 'active_jobs' not in st.session_state or not st.session_state.active_jobs:
        st.info("No jobs to display.")
        return

    for job_id, job_handle in st.session_state.active_jobs.items():
        with st.expander(f"Job: {job_id}", expanded=False):
            status = job_handle.status()
            st.write(f"**Status:** {status}")

            if hasattr(job_handle, 'get_logs'):
                logs = job_handle.get_logs()
                st.text_area("Logs", logs, height=200, key=f"job-logs-{job_id}")
            else:
                st.write("No logs available.")

            if st.button(f"Remove {job_id}"):
                del st.session_state.active_jobs[job_id]
                st.rerun()