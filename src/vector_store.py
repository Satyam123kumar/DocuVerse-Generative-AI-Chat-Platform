from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def create_vector_store(chunks):
    """
    Creates a FAISS vector store from document chunks and saves it locally.
    """
    try:
        # Initialize the embedding model from HuggingFace
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Create the FAISS vector store
        vectorstore = FAISS.from_documents(chunks, embeddings)
        
        # Save the vector store locally
        vectorstore.save_local("faiss_index_store")
        return True
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return False