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

    knowledge_base_id = 'YOUR_KEY_HERE'

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "selected_question" not in st.session_state:
        st.session_state.selected_question = None

    st.markdown("# 🧠 NiChart AI Chatbot")
    st.markdown("### Ask any questions related to NiChart")
    st.markdown("Your responses are not saved on our servers and will be inaccessible if you refresh.")

    st.sidebar.subheader("📜 Chat History")
    for index, entry in enumerate(st.session_state.chat_history):
        if st.sidebar.button(entry['question'], key=f"history_{index}"):
            st.session_state.selected_question = index

    if st.session_state.selected_question is not None:
        selected_entry = st.session_state.chat_history[st.session_state.selected_question]
        with st.expander(f"🗨️ Answer to: {selected_entry['question']}", expanded=True):
            st.markdown(f"**AI:** {selected_entry['answer']}")
            if st.button("❌ Close"):
                st.session_state.selected_question = None

    user_input = st.text_area("Ask your question here (Include the word NiChart if you have a question related to the application):", height=100, max_chars=4000)

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
                        st.write(data["result"])
                        st.info(f"Remaining prompts: {data['remaining_prompts']}")
                    else:
                        data = resp.json()
                        st.error(data.get("error", f"Request failed with status code {resp.status_code}"))
                        st.code(data)
                        if "remaining_prompts" in data:
                            st.info(f"Remaining prompts: {data['remaining_prompts']}")

                    raise ValueError("Placeholder, invocation worked")
                    
                    if "nichart".lower() in user_input.lower():
                        retrieval_response = knowledge_client.retrieve(
                            knowledgeBaseId=knowledge_base_id,
                            retrievalQuery={"text": user_input}
                        )
                        retrieved_docs = [doc["content"]["text"] for doc in retrieval_response["retrievalResults"]]
                        context = "\n".join(retrieved_docs) if retrieved_docs else "No relevant information found."
                    else:
                        context = ""

                    prompt = f"\n\nHuman: {('Answer the question concisely and confidently. Do not hedge your responses. NEVER USE PHRASES LIKE \"BASED ON THE INFORMATION PROVIDED.\" If the context does not answer the question, ignore it and respond with the best possible answer. Using the following information, ' + context + ' answer the question directly: ') if context else ''}{user_input}\n\nAssistant:"

                    body = json.dumps({
                        "prompt": prompt,
                        "max_tokens_to_sample": 300,
                        "temperature": 0.7,
                        "top_p": 0.9,
                    })

                    response = runtime_client.invoke_model(
                        modelId="anthropic.claude-v2:1",
                        contentType="application/json",
                        accept="application/json",
                        body=body
                    )

                    response_body = json.loads(response["body"].read().decode("utf-8"))
                    ai_response = response_body.get("completion", "No response generated.")

                    banned_phrases = [
                        r"\b[bB]ased on the information provided[,]?\b",
                        r"\b[bB]ased on the information provided, [,]?\b",
                        r"\b[aA]ppears to be\b",
                        r"\b[sS]eems like\b",
                        r"\b[fF]rom what I know\b",
                        r"\b[iI]t is possible that\b"
                    ]

                    for phrase in banned_phrases:
                        ai_response = re.sub(phrase, "", ai_response)

                    ai_response = " ".join(ai_response.split())

                    st.session_state.chat_history.append({"question": user_input, "answer": ai_response})

                    st.success("Response:")
                    st.markdown(f"**AI:** {ai_response}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a question.")

utilnav.main_navig("Home", "pages/nichart_home.py", None, None)

# Show session state vars
if st.session_state.mode == 'debug':
    utilses.disp_session_state()
