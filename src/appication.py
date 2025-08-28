import streamlit as st
import faiss
import pickle
import os
import numpy as np
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("❌ GOOGLE_API_KEY not found in environment variables.")
    st.stop()

genai.configure(api_key=api_key)
embedding_model_name = "models/text-embedding-004"
llm_model = genai.GenerativeModel("gemini-1.5-flash")

# -------------------------------
# Load FAISS index & cache
# -------------------------------
INDEX_PATH = "faiss_index.bin"
CACHE_PATH = "cached_embeddings.pkl"
DATA_PATH = "Dataset/bitext_dataset.csv"

if not os.path.exists(INDEX_PATH) or not os.path.exists(CACHE_PATH):
    st.error("❌ FAISS index or cached embeddings not found. Please build them first with model.py")
    st.stop()

index = faiss.read_index(INDEX_PATH)
with open(CACHE_PATH, "rb") as f:
    cached_embeddings = pickle.load(f)

# Load dataset for context responses
df = pd.read_csv(DATA_PATH)
if {"instruction", "response"}.issubset(df.columns):
    df = df.rename(columns={"instruction": "prompt"})
df = df.dropna(subset=["prompt", "response"])

# -------------------------------
# Helper functions
# -------------------------------
def get_embedding(text: str):
    if text in cached_embeddings:
        return cached_embeddings[text]
    response = genai.embed_content(
        model=embedding_model_name,
        content=text,
        task_type="retrieval_document"
    )
    emb = np.array(response["embedding"], dtype=np.float32)
    cached_embeddings[text] = emb
    return emb

def retrieve_context(query, k=3):
    query_vec = get_embedding(query).reshape(1, -1)
    _, indices = index.search(query_vec, k)
    return "\n".join(df.iloc[idx]["response"] for idx in indices[0])

def generate_answer(question, context):
    prompt = f"""
**Customer Support Response Guidelines**
Context Information:
{context}

Current Query:
{question}

Response Requirements:
- Clear, concise answer
- At most 3 sentences
- Use markdown if listing steps
- If unsure, escalate politely
"""
    try:
        response = llm_model.generate_content(prompt)
        return response.text if response and response.text else "Sorry, I couldn’t generate an answer."
    except Exception as e:
        return f"Error: {e}"

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Banking Chatbot", layout="wide")

# ---- Styling ----
st.markdown(
    """
    <style>
    .stApp { 
        background: linear-gradient(135deg, #0f111a 0%, #1f2345 60%, #272b49 100%); 
        color: #ffffff; 
    }
    .stButton>button { background-color: #4863ff; color: white; border-radius: 8px; }
    .stChatMessage, .stChatMessage p, .stChatMessage span, .stChatMessage div {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("💬 Banking Customer Support Chatbot")

# ---- Chat history ----
if "history" not in st.session_state:
    st.session_state.history = []

# Display history
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
query = st.chat_input("Ask your banking question...")
if query:
    st.session_state.history.append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            context = retrieve_context(query)
            response_text = generate_answer(query, context)
            st.markdown(response_text)

    st.session_state.history.append({"role": "assistant", "content": response_text})
