# StudyBot

**StudyBot** is an AI-powered study assistant that lets users upload PDFs and chat with a bot to get detailed answers with explanations. It uses RAG (Retrieval-Augmented Generation) with a vector database to provide context-aware responses.

---

## Features

* User login & registration with MongoDB
* Upload PDFs to build a personal knowledge base
* AI chatbot with detailed, contextual answers
* Stores conversation history per user
* Streamlit web interface for easy interaction

---

## Tech Stack

* **Frontend:** Streamlit
* **Backend:** Python, MongoDB
* **AI Model:** Ollama LLM (Llama 3.2)
* **Embeddings:** HuggingFace `all-MiniLM-L6-v2`
* **Vector DB:** Chroma
* **PDF Loader:** PyPDFLoader
* **Password Hashing:** bcrypt

---

## Quick Start

```bash
git clone <repo_url>
cd studybot
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
streamlit run main.py
```

* Login or register
* Upload PDFs
* Ask questions to StudyBot

---

## Future Enhancements

* Multi-file search and relevance ranking
* Support for more document formats
* Speech-to-text input and audio responses
* Cloud deployment with secure storage
