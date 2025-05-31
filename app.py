import streamlit as st
import json
import openai
from dotenv import load_dotenv
import os
from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load OpenAI API key
openai.api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

# Load chunked aviation content
@st.cache_data
def load_chunks():
    with open("tc_chunks.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

chunks = load_chunks()
chunk_texts = [chunk['content'] for chunk in chunks]

# Embed and search
vectorizer = TfidfVectorizer().fit_transform(chunk_texts)
def search_chunks(query: str, k: int = 5) -> List[str]:
    query_vec = TfidfVectorizer().fit(chunk_texts).transform([query])
    sims = cosine_similarity(query_vec, vectorizer).flatten()
    top_indices = sims.argsort()[-k:][::-1]
    return [chunk_texts[i] for i in top_indices]

# Ask OpenAI to answer based on top chunks
def ask_tutor(question):
    top_chunks = search_chunks(question)
    context = "\n\n".join(top_chunks)
    prompt = f"""
You are an aviation tutor for Canadian PPL students, explaining in simple language.

Context:
{context}

Question: {question}
Answer:"""
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message['content'].strip()

# Streamlit UI
st.title("🇨🇦 PPL Aviation Tutor (Transport Canada)")
st.write("Ask questions about aviation theory and get clear, simple explanations based on Canadian documents.")

question = st.text_input("✈️ Ask a question about aviation...")
if question:
    with st.spinner("Thinking like a flight instructor..."):
        answer = ask_tutor(question)
    st.markdown("### 🧠 Answer")
    st.write(answer)
