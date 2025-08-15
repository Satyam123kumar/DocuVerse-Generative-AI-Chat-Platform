import streamlit as st
import uuid

def initialize_session_state():
    """Initializes the session state variables."""
    if "chats" not in st.session_state:
        st.session_state.chats = {}
    if "current_chat_id" not in st.session_state:
        # Create a default chat session on first run
        chat_id = str(uuid.uuid4())
        st.session_state.current_chat_id = chat_id
        st.session_state.chats[chat_id] = {
            "history": [],
            "chain": None,
            "title": "New Chat",
            "eval_results": None
        }

def switch_chat(chat_id):
    """Switches the current chat session."""
    st.session_state.current_chat_id = chat_id