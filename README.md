# ManPower-CaseStudy
Building a Rag Application using Langchain

This is an intelligent chatbot application that allows you to upload and have a conversation with your PDF documents.
Built with Streamlit and LangChain, it uses Google's Gemini model and a FAISS vector store to provide accurate, context-aware answers.

## ⚙️ How It Works

The application follows a Retrieval-Augmented Generation (RAG) workflow to ensure that the answers are based on factual data from the PDF document

1. **Interactive UI:** Built with Streamlit, the application provides a user-friendly and responsive chat interface, including multi-chat session management.
2. **Advanced Conversational Memory:** It uses a history-aware retriever to understand the context of follow-up questions, allowing for a natural, flowing conversation.
3. **High-Quality Embeddings:** Document and query embeddings are generated using a powerful Hugging Face Sentence Transformer model (sentence-transformers/all-MiniLM-L6-v2).
4. **Efficient Vector Storage:** It leverages FAISS to create a fast, local vector store for efficient semantic search and retrieval.
5. **Verifiable Answers:** Each response from the chatbot includes sources and citations, showing the page number and a snippet from the original PDF used to generate the answer.
6.  **Chat History:** The conversation is stored in memory to handle follow-up questions effectively.

## 🛠️ Tech Stack

* **Language:** Python
* **LLM Framework:** LangChain
* **LLM:** Google Gemini
* **Vector Database:** FAISS
* **Embedding Model:** Hugging Face `sentence-transformers`
* **UI:** Streamlit

## 🚀 Getting Started

Follow these steps to set up and run the project locally.

### Prerequisites

* Git
* Python 3.8 or higher

### Installation & Setup

1.  **Clone the Repository**
    ```sh
    git clone [https://github.com/Satyam123kumar/ManPower-CaseStudy.git]
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

