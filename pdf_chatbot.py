import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter
import faiss
import numpy as np
import openai
import tempfile
import os  

# Page config
st.set_page_config(page_title="PAGE ECHO", layout="centered", page_icon="📄")

# Hide Streamlit UI elements
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Background and title style
st.markdown("""
    <style>
    body {
        background-color: #e6ccff;  /* Soft lilac */
        color: #2c3e50;
    }
    div.stButton > button {
        background-color: #6a0dad;
        color: white;
        border: none;
        padding: 8px 20px;
        border-radius: 6px;
        font-size: 16px;
        margin-top: 32px;
    }
    div.stButton > button:hover {
        background-color: #5c0099;
    }
    </style>
""", unsafe_allow_html=True)

# Title and subtitle
st.markdown("<h1 style='text-align: center; color: #6a0dad;'>🤖 PageEcho</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #333;'>Your Smart PDF Question Answering Assistant</h4>", unsafe_allow_html=True)

# Load API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Welcome Message
if "pdf_uploaded" not in st.session_state:
    st.session_state["pdf_uploaded"] = False

if not st.session_state["pdf_uploaded"]:
    st.info("👋 Welcome! Upload a PDF file to get started.")

# PDF Upload
st.title("📄 Chat with your PDF")
pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])

texts, index, embed_model = [], None, None

if pdf_file and openai.api_key:
    st.session_state["pdf_uploaded"] = True

    with st.spinner("🔄 Processing your PDF, please wait..."):
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(pdf_file.read())
            tmp_path = tmp_file.name

        reader = PdfReader(tmp_path)
        raw_text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                raw_text += content + "\n"

        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = splitter.split_text(raw_text)

        embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = embed_model.encode(texts)

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings))

    st.success("✅ PDF processed. Ask a question below!")

# Question Input with Purple Button
col1, col2 = st.columns([4, 1])
with col1:
    query = st.text_input("Ask a question about the PDF", placeholder="e.g., What is the summary?", key="query_input")
with col2:
    submit = st.button("Search")

# Query Processing
if submit and query and texts and index is not None:
    with st.spinner("💬 Generating answer..."):
        query_embedding = embed_model.encode([query])
        distances, indices = index.search(query_embedding, k=3)
        context = "\n\n".join([texts[i] for i in indices[0]])

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
elif pdf_file and not openai.api_key:
    st.warning("⚠️ No OpenAI API key found. Please add it in Streamlit secrets.")
