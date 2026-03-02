import os
import re
import requests
import streamlit as st
import json

try:
    import boto3
    boto3_available = True
except ImportError:
    boto3_available = False

import utils.utils_menu as utilmenu
import utils.utils_pages as utilpg

from utils.utils_styles import inject_global_css 
import gui.utils_navig as utilnav

from utils.utils_logger import setup_logger
import utils.utils_session as utilses


logger = setup_logger()
logger.debug('Page: Chatbot')

inject_global_css()

# Page config should be called for each page
#utilpg.config_page()
utilpg.set_global_style()

###############################

if 'instantiated' not in st.session_state or not st.session_state.instantiated:
    utilses.init_session_state()


if boto3_available:
    try:
        runtime_client = boto3.client("bedrock-runtime", region_name="us-east-1")
        knowledge_client = boto3.client("bedrock-agent-runtime", region_name="us-east-1")
        chatbot_enabled = True
    except Exception as e:
        chatbot_enabled = False
        error_message = str(e)
else:
    chatbot_enabled = False
    error_message = "Boto3 is not installed."

if not st.session_state.has_cloud_session:
    chatbot_enabled = False
    error_message = "Please use the cloud service (https://cloud.neuroimagingchart.com/) to use the NiChart chatbot."

if not chatbot_enabled:
    st.markdown("# 🚫 Chatbot Service Disabled")
    st.error(f"The AI chatbot service is currently unavailable. Reason: {error_message}")
else:
    API_URL = "https://mdkcwovo4a.execute-api.us-east-1.amazonaws.com/invoke"
    cloud_session_token = st.session_state.cloud_session_token
    if not cloud_session_token:
        st.error("Not authenticated.")
        st.stop()


    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "selected_question" not in st.session_state:
        st.session_state.selected_question = None

    st.markdown("# 🧠 NiChart LLM Chatbot")
    st.markdown("### Ask questions related to NiChart")
    st.markdown("To enhance the quality of the response, please explicitly mention the pipeline you intend to use by name. This will help us retrieve the best information for you.")
    st.markdown("Do not provide any private information to this chatbot. By using this service you confirm that you are authorized to share any information you choose to provide.")
    st.markdown("Please be aware that [LLM responses may be inaccurate or harmful](https://medium.com/ai-for-absolute-beginners/what-is-hallucination-in-ai-b9b5d6eaae73). Always double-check responses.")
    st.markdown("Your questions and responses are not saved on our servers and will be inaccessible if you refresh the page.")


    #st.sidebar.subheader("📜 Chat History")
    #for index, entry in enumerate(st.session_state.chat_history):
    #    if st.sidebar.button(entry['question'], key=f"history_{index}"):
    #        st.session_state.selected_question = index

    #if st.session_state.selected_question is not None:
    #    selected_entry = st.session_state.chat_history[st.session_state.selected_question]
    #    with st.expander(f"🗨️ Answer to: {selected_entry['question']}", expanded=True):
    #        st.markdown(f"**AI:** {selected_entry['answer']}")
    #        if st.button("❌ Close"):
    #            st.session_state.selected_question = None

    user_input = st.text_area("Ask your question here:", height=100, max_chars=4000)

    if st.button("Ask NiChart"):
        if user_input.strip():
            with st.spinner("Thinking..."):
                try:
                    resp = requests.post(
                        API_URL,
                        headers={
                            "Authorization": f"Bearer {cloud_session_token}",
                            "Content-Type": "application/json",
                        },
                        json={"prompt": user_input},
                        timeout=30,
                    )

                    if resp.status_code == 200:
                        data = resp.json()
                        st.markdown("### Response")
                        # Debug
                        #st.write(data["result"])
                        st.write(data["result"]["content"][0]["text"])
                        st.info(f"Remaining prompts: {data['remaining_prompts']}")
                    else:
                        data = resp.json()
                        st.error(data.get("error", f"Request failed with status code {resp.status_code}"))
                        st.code(data)
                        if "remaining_prompts" in data:
                            st.info(f"Remaining prompts: {data['remaining_prompts']}")

                    

                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a question.")

utilnav.main_navig("Home", "pages/nichart_home.py", None, None)

# Show session state vars
if st.session_state.mode == 'debug':
    utilses.disp_session_state()
