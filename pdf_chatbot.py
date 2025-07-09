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

# ========== PAGE CONFIG & STYLING ==========
st.set_page_config(page_title="PageEcho", layout="centered", page_icon="📄")

# Styling and animations for buttons and footer
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

    @keyframes glow {
        0% { box-shadow: 0 0 5px #b266ff, 0 0 10px #b266ff, 0 0 15px #b266ff; }
        50% { box-shadow: 0 0 10px #8a2be2, 0 0 20px #8a2be2, 0 0 30px #8a2be2; }
        100% { box-shadow: 0 0 5px #b266ff, 0 0 10px #b266ff, 0 0 15px #b266ff; }
    }
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-6px); }
    }
    .footer-left-animated {
        position: fixed;
        bottom: 0;
        left: 0;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
        color: white;
        background-color: #6a0dad;
        border-top-right-radius: 12px;
        animation: glow 3s ease-in-out infinite;
        z-index: 9999;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .emoji { animation: bounce 1.5s infinite; font-size: 18px; }
    </style>
""", unsafe_allow_html=True)

# ========== USER LOGIN ==========
USERS = {
    "KRITTIKA GHOSH": "KG@123",
    "SONALI GHOSH": "SG@123"
}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.markdown("""
        <h2 style='color:#6a0dad;'>🔐 Login to 🤖 PageEcho ✨</h2>
        <p style='color:#444;font-size:16px;'>Please enter your credentials to continue.</p>
    """, unsafe_allow_html=True)
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("🔐 Login"):
        if username in USERS and USERS[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("✅ Login successful! Reloading...")
            st.rerun()
        else:
            st.error("❌ Invalid username or password.")
    st.markdown("""
        <div class="footer-left-animated">
            <span class="emoji">👩‍💻</span>
            Created by <b> Krittika Ghosh</b>
        </div>
    """, unsafe_allow_html=True)
    st.stop()

# ========== LOGOUT BUTTON ==========
logout_center = st.columns([4, 1, 4])
with logout_center[1]:
    if st.button("🚪 Logout"):
        for key in st.session_state.keys():
            st.session_state[key] = False
        st.rerun()

# ========== APP HEADER ==========
st.markdown("<h1 style='text-align: center; color: #6a0dad;'>🤖 PageEcho ✨</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #333;'>Where Knowledge Echoes from Every Page 🪄</h4>", unsafe_allow_html=True)

# ========== REMAINING APP CODE GOES HERE ==========
# File upload, processing, embedding, query interface, etc...

# ========== GLOBAL FOOTER ==========
st.markdown("""
    <div class="footer-left-animated">
        <span class="emoji">👩‍💻</span>
        Created by <b> Krittika Ghosh</b>
    </div>
""", unsafe_allow_html=True)

