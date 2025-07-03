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

# Page config
st.set_page_config(page_title="PageEcho", layout="centered", page_icon="📄")

# Hide default Streamlit UI
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Custom styling
st.markdown("""
    <style>
    body {
        background-color: #e6ccff;
        color: #2c3e50;
    }
    div.stButton > button {
        background-color: #6a0dad;
        color: white;
        padding: 10px 30px;
        border-radius: 8px;
        font-size: 18px;
        font-weight: bold;
        transition: all 0.3s ease-in-out;
        animation: pulse 2s infinite;
        white-space: nowrap;
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

# App title
st.markdown("<h1 style='text-align: center; color: #6a0dad;'>🤖 PageEcho ✨</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #333;'>Where Knowledge Echoes from Every Page 🪄</h4>", unsafe_allow_html=True)

# OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Session state setup
if "file_uploaded" not in st.session_state:
    st.session_state["file_uploaded"] = False
    st.session_state["texts"] = []
    st.session_state["index"] = None
    st.session_state["embed_model"] = None
    st.session_state["filename"] = None
    st.session_state["show_processed_msg"] = False

# Upload section
st.markdown("<h3 style='text-align: center; color: #6a0dad;'>🌀 PageEcho Portal: Talk to Your File</h3>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload a file", type=["pdf", "txt", "docx", "xlsx", "csv"])

# Reset state if file removed
if uploaded_file is None:
    st.session_state["file_uploaded"] = False
    st.session_state["texts"] = []
    st.session_state["index"] = None
    st.session_state["embed_model"] = None
    st.session_state["filename"] = None
    st.session_state["show_processed_msg"] = False

# Process uploaded file
if uploaded_file and openai.api_key:
    if uploaded_file.name != st.session_state["filename"]:
        st.session_state["file_uploaded"] = True
        st.session_state["filename"] = uploaded_file.name

        with st.spinner("🔄 Processing your file, please wait..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix="." + uploaded_file.name.split('.')[-1]) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            raw_text = ""
            ext = uploaded_file.name.split('.')[-1].lower()

            try:
                if ext == "pdf":
                    reader = PdfReader(tmp_path)
                    for page in reader.pages:
                        content = page.extract_text()
                        if content:
                            raw_text += content + "\n"
                elif ext == "txt":
                    with open(tmp_path, "r", encoding="utf-8") as f:
                        raw_text = f.read()
                elif ext == "docx":
                    doc = docx.Document(tmp_path)
                    for para in doc.paragraphs:
                        raw_text += para.text + "\n"
                elif ext in ["xlsx", "csv"]:
                    df = pd.read_excel(tmp_path) if ext == "xlsx" else pd.read_csv(tmp_path)
                    raw_text = df.to_string(index=False)
                else:
                    st.error("❌ Unsupported file format.")
                    st.stop()
            except Exception as e:
                st.error(f"Error reading file: {e}")
                st.stop()

            # Process and embed
            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = splitter.split_text(raw_text)

            embed_model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = embed_model.encode(texts)

            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(np.array(embeddings))

            st.session_state["texts"] = texts
            st.session_state["index"] = index
            st.session_state["embed_model"] = embed_model
            st.session_state["show_processed_msg"] = True

if st.session_state["show_processed_msg"]:
    st.success("✅ File processed. Ask a question below!")

# Sample questions
sample_questions = [
    "What is the summary?",
    "List key points discussed.",
    "What is the conclusion?",
    "What are the main findings?",
    "What is the purpose of the document?",
    "Can you explain the methodology?",
    "What are the recommendations?",
    "Summarize the introduction section.",
    "Highlight important dates or events.",
    "What are the challenges mentioned?"
]

# User input
col1, col2 = st.columns([1, 1])
with col1:
    selected_question = st.selectbox("Pick a sample question:", [""] + sample_questions)
with col2:
    custom_question = st.text_input("Or type your question here")

query = custom_question if custom_question else selected_question

# Search button row
button_cols = st.columns([3, 1, 3])
with button_cols[1]:
    submit = st.button("🔍 Search", use_container_width=True)

# Validation and result
if submit:
    if not uploaded_file:
        st.error("⚠️ Please upload a file first.")
    elif not query.strip():
        st.error("⚠️ No question given. Please type or select a question.")
    else:
        st.info(f"🔍 Searching answer for: **{query}**")

        with st.spinner("💬 Generating answer..."):
            query_embedding = st.session_state["embed_model"].encode([query])
            distances, indices = st.session_state["index"].search(query_embedding, k=3)
            context = "\n\n".join([st.session_state["texts"][i] for i in indices[0]])

            prompt = f"""
You are an assistant that answers questions based only on the context below.

Context:
{context}

Question: {query}
Answer:
"""
            try:
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}]
                )
                answer = response.choices[0].message.content.strip()
                st.markdown(f"**Answer:** {answer}")
            except Exception as e:
                st.error(f"❌ OpenAI API error: {e}")

# API key not set
elif uploaded_file and not openai.api_key:
    st.warning("⚠️ No OpenAI API key found. Please add it in Streamlit secrets.")
