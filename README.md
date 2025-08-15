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

