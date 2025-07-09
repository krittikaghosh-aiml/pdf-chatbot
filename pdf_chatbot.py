import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter
import faiss
import numpy as np
import openai
import tempfile
import os
import docx
import pandas as pd

# ========= PAGE CONFIG =========
st.set_page_config(page_title="PageEcho", layout="centered", page_icon="📄")

# ========= USER DATABASE =========
USERS = {
    "KRITTIKA GHOSH": "KG@123",
    "SONALI GHOSH": "SG@123"
}

# ========= SESSION STATE =========
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ========= LOGIN PAGE =========
def login_page():
    st.markdown("<h2 style='text-align: center; color:#6a0dad;'>🔐 Login to 🤖 PageEcho ✨</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Enter your credentials to continue.</p>", unsafe_allow_html=True)
    
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login = st.button("🔐 Login")

    if login:
        if username in USERS and USERS[username] == password:
            st.session_state.logged_in = True
            st.success("✅ Login successful! Please wait...")
            st.rerun()
        else:
            st.error("❌ Invalid username or password.")

# ========= MAIN APP =========
def main_app():
    # Logout
    logout_center = st.columns([4, 1, 4])
    with logout_center[1]:
        if st.button("🚪 Logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # UI Header
    st.markdown("<h1 style='text-align: center; color: #6a0dad;'>🤖 PageEcho ✨</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #333;'>Where Knowledge Echoes from Every Page 🪄</h4>", unsafe_allow_html=True)

    # Styling
    st.markdown("""
        <style>
        #MainMenu, footer, header {visibility: hidden;}
        div.stButton > button {
            background-color: #6a0dad;
            color: white;
            padding: 10px 30px;
            border-radius: 8px;
            font-size: 18px;
            font-weight: bold;
            animation: pulse 2s infinite;
            transition: all 0.3s ease-in-out;
        }
        div.stButton > button:hover {
            background-color: #5c0099;
            transform: scale(1.05);
        }
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(106, 13, 173, 0.5); }
            70% { box-shadow: 0 0 0 10px rgba(106, 13, 173, 0); }
            100% { box-shadow: 0 0 0 0 rgba(106, 13, 173, 0); }
        }
        </style>
    """, unsafe_allow_html=True)

    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Initialize session vars
    for key in ["file_uploaded", "texts", "index", "embed_model", "filename", "show_processed_msg"]:
        if key not in st.session_state:
            st.session_state[key] = False if key == "file_uploaded" else None

    # File upload
    st.markdown("<h3 style='text-align: center; color: #6a0dad;'>🌀 PageEcho Portal: Talk to Your File</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a file", type=["pdf", "txt", "docx", "xlsx", "csv"])

    if uploaded_file and openai.api_key:
        if uploaded_file.name != st.session_state["filename"]:
            st.session_state["file_uploaded"] = True
            st.session_state["filename"] = uploaded_file.name

            with st.spinner("🔄 Processing your file..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix="." + uploaded_file.name.split('.')[-1]) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                ext = uploaded_file.name.split('.')[-1].lower()
                raw_text = ""

                try:
                    if ext == "pdf":
                        reader = PdfReader(tmp_path)
                        raw_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
                    elif ext == "txt":
                        with open(tmp_path, "r", encoding="utf-8") as f:
                            raw_text = f.read()
                    elif ext == "docx":
                        doc = docx.Document(tmp_path)
                        raw_text = "\n".join([para.text for para in doc.paragraphs])
                    elif ext in ["xlsx", "csv"]:
                        df = pd.read_excel(tmp_path) if ext == "xlsx" else pd.read_csv(tmp_path)
                        raw_text = df.to_string(index=False)
                    else:
                        st.error("❌ Unsupported file format.")
                        st.stop()
                except Exception as e:
                    st.error(f"❌ Error reading file: {e}")
                    st.stop()

                splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                texts = splitter.split_text(raw_text)

                embed_model = SentenceTransformer('all-MiniLM-L6-v2')
                embeddings = embed_model.encode(texts)

                index = faiss.IndexFlatL2(embeddings.shape[1])
                index.add(np.array(embeddings))

                st.session_state.update({
                    "texts": texts,
                    "index": index,
                    "embed_model": embed_model,
                    "show_processed_msg": True
                })

    if st.session_state["show_processed_msg"]:
        st.success("✅ File processed. Ask a question below!")

        # Questions
        sample_questions = [
            "What is the summary?",
            "What are the main findings?",
            "What is the purpose of the document?",
            "What are the recommendations?",
        ]

        col1, col2 = st.columns(2)
        with col1:
            selected_question = st.selectbox("Sample Question", [""] + sample_questions)
        with col2:
            custom_question = st.text_input("Or type your question")

        query = custom_question if custom_question else selected_question

        if st.button("🔍 Search"):
            if not query.strip():
                st.error("⚠️ Please ask a question.")
            else:
                with st.spinner("💬 Generating answer..."):
                    query_embedding = st.session_state["embed_model"].encode([query])
                    distances, indices = st.session_state["index"].search(query_embedding, k=3)
                    context = "\n\n".join([st.session_state["texts"][i] for i in indices[0]])

                    prompt = f"""You are an assistant that answers based only on the context below.

Context:
{context}

Question: {query}
Answer:"""

                    try:
                        response = openai.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": prompt}]
                        )
                        answer = response.choices[0].message.content.strip()
                        st.markdown(f"**Answer:** {answer}")
                    except Exception as e:
                        st.error(f"❌ OpenAI Error: {e}")

# ========= RENDER APP =========
if not st.session_state.logged_in:
    login_page()
else:
    main_app()
