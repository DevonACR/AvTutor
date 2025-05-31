import streamlit as st
import json
import openai
import os
from typing import List
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import random

# Load secrets and API key
load_dotenv()
openai.api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

# Load aviation content
@st.cache_data
def load_chunks():
    with open("tc_chunks.json", "r", encoding="utf-8") as f:
        return json.load(f)

chunks = load_chunks()
chunk_texts = [chunk['content'] for chunk in chunks]

# Embed all chunks
vectorizer = TfidfVectorizer().fit_transform(chunk_texts)

def search_chunks(query: str, k: int = 5) -> List[str]:
    query_vec = TfidfVectorizer().fit(chunk_texts).transform([query])
    sims = cosine_similarity(query_vec, vectorizer).flatten()
    top_indices = sims.argsort()[-k:][::-1]
    return [chunk_texts[i] for i in top_indices]

# Ask a question
def ask_tutor(question):
    top_chunks = search_chunks(question)
    context = "\n\n".join(top_chunks)
    prompt = f"""
You are an aviation tutor for Canadian PPL students. Explain clearly and simply.

Context:
{context}

Question: {question}
Answer:"""
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()

# Generate quiz question
def generate_question():
    topic = random.choice(chunk_texts)
    prompt = f"""
Read the following study material and create a 4-option multiple-choice question (A, B, C, D).
Clearly mark the correct answer by its letter.

Study Content:
{topic}

Format your output as:
Question: <text>
A: <choice>
B: <choice>
C: <choice>
D: <choice>
Correct Answer: <A/B/C/D>
"""
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    content = response.choices[0].message.content.strip()

    lines = content.split("\n")
    question = lines[0].replace("Question: ", "").strip()
    choices = {}
    for line in lines[1:5]:
        letter, choice = line.split(":", 1)
        choices[letter.strip()] = choice.strip()
    correct = lines[-1].split(":")[-1].strip()
    return {"question": question, "choices": choices, "correct_answer": correct}

# Explain a topic
def explain_topic(topic):
    top_chunks = search_chunks(topic)
    context = "\n\n".join(top_chunks)
    prompt = f"""
You are an aviation tutor for Canadian PPL students. Explain this topic clearly and simply.

Topic: {topic}

Relevant Info:
{context}

Explanation:"""
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    return response.choices[0].message.content.strip()

# Study categories (manual list for now)
categories = {
    "Air Law": "air law regulations procedures ATC rules",
    "Navigation": "VFR navigation charts headings wind correction",
    "Meteorology": "weather clouds fog icing METAR TAF",
    "Aircraft Operations": "performance speeds loading",
    "Human Factors": "fatigue hypoxia illusions",
    "Radio & Comms": "radio procedures phraseology",
}

# ------------------- Streamlit UI -------------------

st.title("🇨🇦 PPL Aviation Tutor")

mode = st.sidebar.radio("Choose Study Mode", ["🔎 Ask a Question", "🧠 Quiz Me", "🧾 Explain a Topic", "📚 Study by Category"])

if mode == "🔎 Ask a Question":
    question = st.text_input("Ask anything about PPL aviation...")
    if question:
        with st.spinner("Thinking like a flight instructor..."):
            answer = ask_tutor(question)
        st.markdown("### 🧠 Answer")
        st.write(answer)

elif mode == "🧠 Quiz Me":
    st.subheader("🧠 Aviation Quiz")

    if "quiz" not in st.session_state:
        st.session_state.quiz = [generate_question() for _ in range(5)]
        st.session_state.current_q = 0
        st.session_state.submitted = [False] * 5
        st.session_state.answers = [None] * 5
        st.session_state.score = 0

    quiz = st.session_state.quiz
    current_q = st.session_state.current_q
    question_data = quiz[current_q]

    st.markdown(f"**Q{current_q + 1}:** {question_data['question']}")
    user_answer = st.radio(
        "Select your answer:",
        options=["A", "B", "C", "D"],
        format_func=lambda x: f"{x}: {question_data['choices'][x]}",
        index=["A", "B", "C", "D"].index(st.session_state.answers[current_q]) if st.session_state.answers[current_q] else 0,
        key=f"q{current_q}_answer"
    )

    st.session_state.answers[current_q] = user_answer

    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if st.button("⬅️ Previous"):
            if current_q > 0:
                st.session_state.current_q -= 1

    with col3:
        if st.button("Next ➡️"):
            if current_q < len(quiz) - 1:
                st.session_state.current_q += 1

    with col2:
        if not st.session_state.submitted[current_q]:
            if st.button("Submit Answer"):
                correct = question_data['correct_answer']
                if user_answer == correct:
                    st.session_state.score += 1
                st.session_state.submitted[current_q] = True

    # Show feedback
    if st.session_state.submitted[current_q]:
        correct = question_data['correct_answer']
        if user_answer == correct:
            st.success("✅ Correct!")
        else:
            st.error(f"❌ Incorrect. Correct answer: {correct}: {question_data['choices'][correct]}")

    st.markdown(f"**Progress:** {sum(st.session_state.submitted)} / {len(quiz)} answered")
    if all(st.session_state.submitted):
        st.success(f"🎉 Quiz complete! Your score: {st.session_state.score}/{len(quiz)}")


elif mode == "🧾 Explain a Topic":
    topic = st.text_input("Enter a topic you'd like explained (e.g., Class C Airspace)")
    if topic:
        with st.spinner("Reviewing the material..."):
            explanation = explain_topic(topic)
        st.markdown("### 📘 Explanation")
        st.write(explanation)

elif mode == "📚 Study by Category":
    category = st.selectbox("Choose a category", list(categories.keys()))
    if category:
        with st.spinner(f"Reviewing {category}..."):
            explanation = explain_topic(categories[category])
        st.markdown(f"### 📘 {category} Summary")
        st.write(explanation)
