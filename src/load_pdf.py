import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def process_pdf(pdf_file):
    """
    1. Loads a PDF
    2. splits it into chunks using RecursiveCharacterTextSplitter
    3. Returns the chunks
    """
    try:
        # Save the uploaded file temporarily to be read by PyPDFLoader
        with open(pdf_file.name, "wb") as f:
            f.write(pdf_file.getbuffer())
        
        loader = PyPDFLoader(pdf_file.name)
        documents = loader.load()
        
        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
        chunks = text_splitter.split_documents(documents)
        
        # Clean up the temporary file
        os.remove(pdf_file.name)
        
        return chunks
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return None