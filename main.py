import streamlit as st
import os
import pymongo
from datetime import datetime ,timezone

from langchain_ollama import OllamaLLM
from langchain.chains import ConversationalRetrievalChain
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
import bcrypt

# -----------------------------
# MongoDB setup
# -----------------------------
MONGO_URI = "mongodb://localhost:27017/"
client = pymongo.MongoClient(MONGO_URI)
db = client["studybot_db"]
chat_collection = db["chat_history"]
users_collection = db["users"]

# -----------------------------
# Initialize LLM & embeddings
# -----------------------------
llm = OllamaLLM(model="llama3.2:latest")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# Base directory for Chroma vector DBs
BASE_CHROMA_DIR = "chroma_users"
os.makedirs(BASE_CHROMA_DIR, exist_ok=True)

# -----------------------------
# Helper functions
# -----------------------------
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed)

def register_user(username, password):
    if users_collection.find_one({"username": username}):
        return False
    hashed = hash_password(password)
    users_collection.insert_one({"username": username, "password": hashed})
    return True

def login_user(username, password):
    user = users_collection.find_one({"username": username})
    if user and check_password(password, user["password"]):
        return True
    return False

def save_message(user_id, role, content):
    chat_collection.insert_one({
        "user_id": user_id,
        "role": role,
        "content": content,
        "timestamp": datetime.now(timezone.utc)
    })

def load_chat_history(user_id):
    messages = chat_collection.find({"user_id": user_id}).sort("timestamp", 1)
    return [(msg["role"], msg["content"]) for msg in messages]

def get_user_vectordb(user_id):
    user_dir = os.path.join(BASE_CHROMA_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)
    return Chroma(
        persist_directory=user_dir,
        embedding_function=embeddings
    )

def add_pdf_to_chroma(user_id, uploaded_file):
    temp_path = os.path.join("temp_files", uploaded_file.name)
    os.makedirs("temp_files", exist_ok=True)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    loader = PyPDFLoader(temp_path)
    docs = loader.load_and_split()
    
    vectordb = get_user_vectordb(user_id)
    vectordb.add_documents(docs)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="StudyBot", layout="wide")
st.title("StudyBot")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_id" not in st.session_state:
    st.session_state.user_id = ""

# -----------------------------
# Login / Register
# -----------------------------
if not st.session_state.logged_in:
    st.subheader("Login or Register")
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Login"):
            if login_user(username, password):
                st.session_state.logged_in = True
                st.session_state.user_id = username
                st.success("Logged in successfully")
            else:
                st.error("Invalid credentials")
    with col2:
        if st.button("Register"):
            if register_user(username, password):
                st.success("Registered successfully! You can now log in")
            else:
                st.error("Username already exists")

else:
    st.subheader(f"Welcome, {st.session_state.user_id}")
    user_id = st.session_state.user_id
    vectordb = get_user_vectordb(user_id)

    # -----------------------------
    # Upload PDFs
    # -----------------------------
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for file in uploaded_files:
            add_pdf_to_chroma(user_id, file)
        st.success("PDFs added to your personal knowledge base")

    # -----------------------------
    # Chat with StudyBot (Streamlit Form)
    # -----------------------------
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = load_chat_history(user_id)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
        memory=memory
    )

    # Optional: structured prompt template
    template = """You are a helpful study assistant.
Use the following chat history and the latest question to answer in detail explaining the concept.
not only give the answer but also explain the reasoning behind it.then provide examples if applicable.
Chat History:
{history}

Question:
{question}

Answer:"""
    chat_prompt = ChatPromptTemplate.from_template(template)

    # Chat input form with auto-clear
    with st.form("chat_form", clear_on_submit=True):
        user_query = st.text_input("Ask a question:")
        submitted = st.form_submit_button("Send")

        if submitted and user_query:
            response = qa_chain({"question": user_query})
            save_message(user_id, "You", user_query)
            save_message(user_id, "Bot", response["answer"])
            st.session_state.chat_history.append(("You", user_query))
            st.session_state.chat_history.append(("Bot", response["answer"]))

    # Display chat history
    for role, msg in st.session_state.chat_history:
        st.markdown(f"**{role}:** {msg}")
