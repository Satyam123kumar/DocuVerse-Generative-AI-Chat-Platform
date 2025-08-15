import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

from src.load_pdf import process_pdf
from src.vector_store import create_vector_store
from src.chain import create_conversational_chain

from src.evaluation import get_evaluation_score, EVALUATION_QUESTIONS
from src.new_chat import initialize_session_state, switch_chat

import uuid
import pandas as pd

def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(page_title="Chat with your PDF", page_icon="📄")
    
    initialize_session_state()

    # --- Sidebar ---
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("Enter your Google Gemini API Key:", type="password")
        
        uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")
        
        process_button = st.button("Process Document")

        st.divider()
        st.header("Your Chats")

        # New Chat button
        if st.button("➕ New Chat"):
            chat_id = str(uuid.uuid4())
            st.session_state.chats[chat_id] = {
                "history": [],
                "chain": None,
                "title": "New Chat",
                "eval_results": None
            }
            switch_chat(chat_id)
            st.rerun()

        # Display list of previous chats
        for chat_id, chat_data in st.session_state.chats.items():
            if st.button(chat_data["title"], key=chat_id, use_container_width=True):
                switch_chat(chat_id)
                st.rerun()
        
        st.divider()
        # --- Evaluation Section ---
        st.header("Evaluation")
        if st.button("Run Basic Evaluation"):
            current_chat = st.session_state.chats[st.session_state.current_chat_id]
            if current_chat["chain"]:
                if api_key:
                    with st.spinner("Running evaluation..."):
                        eval_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)
                        eval_results = []
                        for item in EVALUATION_QUESTIONS:
                            response = current_chat["chain"].invoke({
                                "input": item["question"],
                                "chat_history": [] # Use empty history for clean evaluation
                            })
                            generated_answer = response.get("answer", "N/A")
                            score, justification = get_evaluation_score(
                                item["question"],
                                generated_answer,
                                item["ground_truth"],
                                eval_llm
                            )
                            eval_results.append({
                                "Question": item["question"],
                                "Generated Answer": generated_answer,
                                "Ground Truth": item["ground_truth"],
                                "Score": score,
                                "Justification": justification
                            })
                        current_chat["eval_results"] = pd.DataFrame(eval_results)
                        st.rerun()
                else:
                    st.warning("Please enter your API key to run the evaluation.")
            else:
                st.warning("Please process a document first to run an evaluation.")


    # --- Main Chat Interface ---
    current_chat = st.session_state.chats[st.session_state.current_chat_id]
    st.title(current_chat["title"])
    st.markdown("Upload a PDF and start asking questions!")

    # Handle document processing
    if process_button:
        if not api_key:
            st.error("Please enter your Gemini API key.")
        elif not uploaded_file:
            st.error("Please upload a PDF document.")
        else:
            with st.spinner("Processing your document... This may take a moment."):
                chunks = process_pdf(uploaded_file)
                if chunks:
                    if create_vector_store(chunks):
                        # Create a new chat for the processed document
                        chat_id = str(uuid.uuid4())
                        st.session_state.chats[chat_id] = {
                            "history": [],
                            "chain": create_conversational_chain(api_key),
                            "title": uploaded_file.name,
                            "eval_results": None
                        }
                        switch_chat(chat_id)
                        st.success("Document processed! Started a new chat.")
                        st.rerun()
                    else:
                        st.error("Failed to create the vector store.")
                else:
                    st.error("Failed to process the PDF document.")

    # Display evaluation results if they exist
    if current_chat["eval_results"] is not None:
        st.subheader("Evaluation Results")
        st.dataframe(current_chat["eval_results"], use_container_width=True)
        st.info("Scores are generated by an AI evaluator (1=Poor, 10=Excellent).")

    # Display chat history from the current chat session
    for message in current_chat['history']:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
            if message['role'] == 'ai' and 'sources' in message:
                with st.expander("View Sources"):
                    for source in message['sources']:
                        st.info(f"Page {source['page']}: {source['snippet']}")


    # Handle user input
    user_query = st.chat_input("Ask a question about your document...")
    if user_query:
        if current_chat["chain"]:
            # Update chat title if it's the first message
            if current_chat["title"] == "New Chat":
                current_chat["title"] = user_query[:30] + "..."

            current_chat['history'].append({"role": "user", "content": user_query})

            # Prepare chat history for the chain
            history_for_chain = []
            for msg in current_chat['history'][:-1]: # Exclude the last user message
                if msg['role'] == 'user':
                    history_for_chain.append(HumanMessage(content=msg['content']))
                elif msg['role'] == 'ai':
                    history_for_chain.append(AIMessage(content=msg['content']))
            
            with st.spinner("Thinking..."):
                response = current_chat["chain"].invoke({
                    "input": user_query,
                    "chat_history": history_for_chain
                })
                
                ai_response = response.get("answer", "Sorry, I couldn't find an answer.")
                
                sources = []
                if "context" in response:
                    for doc in response["context"]:
                        sources.append({
                            "page": doc.metadata.get('page', 'N/A'),
                            "snippet": doc.page_content[:200] + "..."
                        })
                
                current_chat['history'].append({"role": "ai", "content": ai_response, "sources": sources})
                st.rerun()
        else:
            st.warning("Please process a document first to start a conversation.")

if __name__ == "__main__":
    main()