# ManPower-CaseStudy
Building a Rag Application using Langchain

This is an intelligent chatbot application that allows you to upload and have a conversation with your PDF documents. Built with Streamlit and LangChain, it uses Google's Gemini model and a FAISS vector store to provide accurate, context-aware answers with source citations, and it uses HuggingFace to generate embeddings. 

## ⚙️ Core Features

The application follows a Retrieval-Augmented Generation (RAG) workflow to ensure that the answers are based on factual data from the PDF document

1. **Interactive UI:** Built with Streamlit, the application provides a user-friendly and responsive chat interface.
2. **Multi-Chat Management:** Start new conversations or switch between previous chat sessions, all managed in the sidebar.
3. **Conversational Memory:** The chatbot understands the context of follow-up questions for a natural, flowing conversation.
4. **Verifiable Answers:** Each response includes citations, showing the page number and a snippet from the original PDF.
5. **Automated Evaluation:** An integrated evaluation system uses an "LLM-as-a-judge" to score the chatbot's accuracy against a set of test questions.
6. **Modular Codebase:** The code is structured into separate modules for the UI (app.py) and core logic in the src folder, making it easy to maintain and extend.

## 🛠️ Technology Deep Dive

The application follows the RAG workflow to ensure that answers are based on factual data from the PDF document.
1. **Text Splitting: RecursiveCharacterTextSplitter**
   
    This application uses the RecursiveCharacterTextSplitter from LangChain. It tries to split the text based on a hierarchy of separators (e.g., paragraphs, then sentences, then words). This method helps to keep semantically related pieces of text together, which is crucial for maintaining context.

2. **Embedding Model: sentence-transformers/all-MiniLM-L6-v2**

    This application uses the sentence-transformers/all-MiniLM-L6-v2 model from the Hugging Face Sentence Transformers library.
    * **High Performance:** It's a lightweight yet powerful model that excels at creating meaningful semantic embeddings, and it can run locally. 
    * **Dimension:** It transforms each text chunk into a 384-dimensional vector.
    
3. **Vector Store: FAISS**

    Once the text chunks are converted into embeddings, they need to be stored in a specialized database that can perform fast similarity searches. This project uses FAISS (Facebook AI Similarity Search).

4. **Retrieval Technique: A Multi-faceted Approach**

    This application has been designed to leverage several advanced retrieval strategies to ensure the highest quality context is provided to the LLM. I have experimented with and combined the strengths of    **Semantic Similarity**, **Maximum Marginal Relevance (MMR)**, and **Contextual Compression**.
    * **Semantic Similarity (The Foundation):** This is the primary retrieval method. It finds document chunks whose vector embeddings are mathematically closest to the query's embedding. 
    * **Maximum Marginal Relevance (MMR) (For Diversity):** It enhances the retrieval process by balancing two factors: the relevance of a document to the query and the diversity of context. 
    * **Contextual Compression (For Precision):** This application's final and most advanced retrieval step uses a ContextualCompressionRetriever. After an initial, broader search for relevant documents, this technique uses the Gemini LLM to go through the retrieved chunks and "compress" them by extracting only the most relevant sentences and phrases. This is good because it removes noise and irrelevant information.
7. **Conversational Memory: History-Aware Retriever**

    To handle follow-up questions, the application uses a create_history_aware_retriever. Before searching for documents, this retriever first uses the LLM to look at the chat history and the latest user question. It then rephrases the user's input into a standalone question that contains all the necessary context from the conversation.
8. **Language Model (LLM): Google Gemini**

    All generative tasks in this application, like rephrasing questions, generating final answers, and evaluating accuracy, are powered by Google's Gemini model.
9. **Automated Evaluation**

    To ensure the chatbot's reliability, an automated evaluation feature is included. It uses an "LLM-as-a-judge" approach. In this, the system runs a set of predefined questions against the document. For each question, it compares the chatbot's generated answer to a "ground truth" answer. The Gemini model acts as an impartial judge, providing a score from 1 to 10 and a brief justification for the chatbot's performance on each question. This provides a quick and objective measure of the RAG pipeline's accuracy.

## 🚀 Getting Started

Follow these steps to set up and run the project locally.

### Prerequisites

* Git
* Python 3.8 or higher

### Installation & Setup

1.  **Clone the Repository**
    ```sh
    git clone [https://github.com/Satyam123kumar/ManPower-CaseStudy.git]
    cd ManPower-CaseStudy
    ```

2.  **Create a Virtual Environment**
    ```sh
    python -m venv mpgenv
    ```
    Activate the environment:
    * On Windows:
        ```sh
        mpgenv\scripts\activate
        ```
    * On macOS/Linux:
        ```sh
        source mpgenv/bin/activate
        ```

3.  **Install Dependencies**
    ```sh
    pip install -r requirements.txt
    ```

4.  **Run the Application**
    ```sh
    streamlit app.py
    ```

7.  **Open the Chatbot**
    Navigate to `[http://172.16.0.2:8501]` in your web browser.

## License

Distributed under the MIT License. See `LICENSE` for more information.

