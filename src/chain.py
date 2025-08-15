from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.retrievers.document_compressors import LLMChainExtractor
# from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever

def create_conversational_chain(api_key):
    """
    Creates and returns a conversational RAG chain with memory
    """
    try:
        # Initialize the LLM
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)
        
        # Initialize the embedding model
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Load the local FAISS vector store
        vectorstore = FAISS.load_local("faiss_index_store", embeddings, allow_dangerous_deserialization=True
                                       )
        
        # Create a retriever from the vector store
        # 1. Create your base retriever from the vector store
        retriever = vectorstore.as_retriever(search_type="similarity")

        # retriever = MultiQueryRetriever.from_llm(
        #     retriever=base_retriever, 
        #     llm=llm
        # )

        # Create a compressor for the retriever
        # This compressor uses the LLM to extract relevant information from the retrieved documents
        # compressor = LLMChainExtractor(llm)
        # retriever = ContextualCompressionRetriever(
        #     base_retriever=base_retriever,
        #     base_compressor=compressor,
        # )
        
        # Create a history-aware retriever
        # This prompt helps the LLM rephrase a follow-up question to be a standalone question
        history_aware_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
        ])
        
        history_aware_retriever_chain = create_history_aware_retriever(
            llm, retriever, history_aware_prompt
        )
        
        # --- Create the main question-answering chain ---
        # This prompt is for the final answer generation
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant for question-answering tasks. Answer the user's questions based on the below context:\n\n{context}."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])
        
        #Augment the question-answering chain with the context
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        
        # --- Create the final RAG chain ---
        # This chain combines the history-aware retriever and the QA chain
        rag_chain = create_retrieval_chain(history_aware_retriever_chain, question_answer_chain)
        
        return rag_chain
    except Exception as e:
        print(f"Error creating conversational chain: {e}")
        return None
