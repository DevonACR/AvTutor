import streamlit as st
import json
import openai
import os
from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Load OpenAI API key from Streamlit secrets or environment
openai.api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

# Load chunked aviation content
@st.cache_data(show_spinner=False)
def load_chunks():
    with open("tc_chunks.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

chunks = load_chunks()
chunk_texts = [chunk['content'] for chunk in chunks]

# Cache vectorizer and matrix for performance
@st.cache_data(show_spinner=False)
def get_vectorizer_and_matrix(texts):
    vectorizer = TfidfVectorizer().fit(texts)
    matrix = vectorizer.transform(texts)
    return vectorizer, matrix

vectorizer, matrix = get_vectorizer_and_matrix(chunk_texts)

def search_chunks(query: str, k: int = 5) -> List[str]:
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, matrix).flatten()
    top_indices = sims.argsort()[-k:][::-1]
    return [chunk_texts[i] for i in top_indices]

def ask_tutor(question):
    top_chunks = search_chunks(question)
    context = "\n\n".join(top_chunks)
    prompt = f"""
You are an aviation tutor for Canadian PPL students, explaining in simple language.

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

def generate_quiz_questions(topic, n=5):
    prompt = f"""
Generate {n} multiple choice questions for a Private Pilot License student on the topic: {topic}.  
Provide each question with 4 answer choices labeled A, B, C, D. Indicate the correct answer after each question.
"""
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

def parse_quiz(text):
    # Basic parser to split questions and answers
    questions = re.split(r"\n\s*\d+\.\s", text)
    parsed = []
    for q in questions[1:]:  # skip before first question number
        lines = q.strip().split('\n')
        question_text = lines[0]
        choices = {}
        answer = None
        for line in lines[1:]:
            m = re.match(r"([A-D])\.\s*(.*)", line)
            if m:
                choices[m.group(1)] = m.group(2)
            elif line.lower().startswith("correct answer:"):
                answer = line.split(":")[-1].strip()
        parsed.append({'question': question_text, 'choices': choices, 'answer': answer})
    return parsed

def explain_topic(topic, chunks, max_chunks=5):
    relevant_chunks = [chunk['content'] for chunk in chunks if topic.lower() in chunk['content'].lower()]
    if not relevant_chunks:
        return "Sorry, no relevant content found for this topic."
    combined_text = "\n\n".join(relevant_chunks[:max_chunks])

    prompt = f"""
Explain the following content about {topic} in simple terms for a student:

{combined_text}
"""
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()

# Extract categories from chunks dynamically
all_categories = sorted(set(chunk.get('category', 'Uncategorized') for chunk in chunks))

st.title("🇨🇦 PPL Aviation Tutor (Transport Canada)")

mode = st.sidebar.radio(
    "Select Study Mode",
    ['Ask a Question', 'Quiz Me', 'Explain a Topic', 'Study by Category']
)

if mode == 'Ask a Question':
    st.write("Ask questions about aviation theory and get clear, simple explanations based on Canadian documents.")
    question = st.text_input("✈️ Ask a question about aviation...")
    if question:
        with st.spinner("Thinking like a flight instructor..."):
            answer = ask_tutor(question)
        st.markdown("### 🧠 Answer")
        st.write(answer)

elif mode == 'Quiz Me':
    st.write("Generate multiple choice quiz questions on a topic.")
    topic = st.text_input("Enter topic for quiz questions:")
    n = st.slider("Number of questions", 1, 10, 5)

    if topic and st.button("Generate Quiz"):
        with st.spinner("Generating quiz questions..."):
            quiz_text = generate_quiz_questions(topic, n)
            quiz = parse_quiz(quiz_text)
            st.session_state['quiz'] = quiz
            st.session_state['answers'] = [None] * len(quiz)
            st.session_state['submitted'] = False

    if 'quiz' in st.session_state and st.session_state['quiz']:
        st.markdown("### Quiz")
        for i, q in enumerate(st.session_state['quiz'], 1):
            st.markdown(f"**Q{i}. {q['question']}**")
            options = list(q['choices'].items())  # List of tuples (A, choice text)
            selected = st.radio(
                label=f"Select answer for question {i}",
                options=[opt[0] for opt in options],
                format_func=lambda x: f"{x}. {q['choices'][x]}",
                key=f"q{i}"
            )
            st.session_state['answers'][i-1] = selected

        if st.button("Submit Answers"):
            st.session_state['submitted'] = True

        if st.session_state.get('submitted', False):
            score = 0
            for i, q in enumerate(st.session_state['quiz']):
                correct = q['answer']
                user_ans = st.session_state['answers'][i]
                is_correct = user_ans == correct
                if is_correct:
                    score += 1
                st.markdown(
                    f"Q{i+1} Correct answer: **{correct}** | Your answer: **{user_ans}** "
                    + ("✅" if is_correct else "❌")
                )
            st.markdown(f"### Your score: {score} / {len(st.session_state['quiz'])}")


elif mode == 'Explain a Topic':
    st.write("Get a simple explanation of a topic based on your study material.")
    topic = st.text_input("Enter topic to explain:")
    if topic:
        if st.button("Explain"):
            with st.spinner(f"Explaining {topic}..."):
                explanation = explain_topic(topic, chunks)
                st.markdown("### Explanation")
                st.write(explanation)

elif mode == 'Study by Category':
    st.write("Browse content by category.")
    category = st.selectbox("Select category:", all_categories)
    category_chunks = [chunk['content'] for chunk in chunks if chunk.get('category', 'Uncategorized') == category]
    st.write(f"Showing {len(category_chunks)} chunks in category **{category}**:")
    for text in category_chunks:
        st.write(text)
