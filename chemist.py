import warnings
import logging
import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Disable warnings and info logs
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# --------------------- Custom CSS for Bio Lab Theme ---------------------
st.markdown("""
    <style>
    body {
        background-color: #0d1b2a;
        color: #e0e1dd;
    }
    .stApp {
        background-color: #0d1b2a;
        font-family: 'Segoe UI', sans-serif;
    }
    h1 {
        text-align: center;
        color: #00ffcc;
        font-weight: bold;
        text-shadow: 0 0 10px #00ffcc;
    }
    .chat-message {
        background-color: #1b263b;
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0px 0px 10px rgba(0, 255, 204, 0.3);
    }
    .stChatInput input {
        background-color: #1b263b;
        color: #e0e1dd;
        border: 1px solid #00ffcc;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title('🧠 Toxiscan: one search for any Unknown Ingredients in your Daily Life')

# Setup a session state variable to hold all the old messages
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display all the historical messages
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(f"<div class='chat-message'>{message['content']}</div>", unsafe_allow_html=True)

prompt = st.chat_input('🔬 Paste your chemical ingredients here...')

if prompt:
    with st.chat_message('user'):
        st.markdown(f"<div class='chat-message'>{prompt}</div>", unsafe_allow_html=True)
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    groq_syspromt = PromptTemplate(
        input_variables=["user_prompt"],
        template="""You are a famous scientist like Andrew Huberman who has an expert knowledge of chemicals and their effects on 
        the human endocrine system. You are a chemist and you are very good at explaining the effects of chemicals on the human body.
        You will be provided a list of ingredients, you will have to detect the endocrine disrupting chemicals and their effects on the human body
        one by one. Here is the ingredient list {user_prompt}."""
    )

    groq_chat = ChatGroq(
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        model_name="mistral-saba-24b"
    )

    Chain = groq_syspromt | groq_chat | StrOutputParser()
    response = Chain.invoke({"user_prompt": prompt})

    with st.chat_message('assistant'):
        st.markdown(f"<div class='chat-message'>{response}</div>", unsafe_allow_html=True)
    st.session_state.messages.append({'role': 'assistant', 'content': response})
