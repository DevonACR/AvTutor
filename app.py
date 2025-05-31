import streamlit as st
import json
import openai
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

# Vectorizer & embeddings for search
vectorizer = TfidfVectorizer().fit_transform(chunk_texts)

def search_chunks(query: str, k: int = 5) -> List[str]:
    query_vec = TfidfVectorizer().fit(chunk_texts).transform([query])
    sims = cosine_similarity(query_vec, vectorizer).flatten()
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

# --- New Quiz Generation & Parsing Functions ---

def generate_quiz_questions(topic: str, num_questions: int = 5):
    prompt = f"""
You are an expert aviation instructor creating a quiz for Canadian Private Pilot License (PPL) students. 
Generate {num_questions} multiple-choice questions on the topic "{topic}".
Each question should have 4 answer choices labeled A, B, C, and D.
Indicate the correct answer explicitly.
Return the quiz in valid JSON format as a list of objects with fields: question, choices (dict with keys A-D), correct_answer (A/B/C/D).

Example:
[
  {{
    "question": "What is the minimum visibility required for VFR flight?",
    "choices": {{
      "A": "3 statute miles",
      "B": "1 statute mile",
      "C": "5 statute miles",
      "D": "10 statute miles"
    }},
    "correct_answer": "A"
  }},
  ...
]

Begin:
"""
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    return response.choices[0].message.content.strip()

def parse_quiz(raw_quiz_text):
    import json
    try:
        quiz = json.loads(raw_quiz_text)
        # Validate basic structure
        for q in quiz:
            assert "question" in q and "choices" in q and "correct_answer" in q
        return quiz
    except Exception as e:
        st.error(f"Error parsing quiz JSON: {e}")
        return []

# --- Explain a Topic (chunk + summarize) ---

def explain_topic(topic):
    top_chunks = search_chunks(topic, k=5)
    context = "\n\n".join(top_chunks)
    prompt = f"""
You are an aviation tutor for Canadian PPL students. Explain the following topic clearly and simply, using the context below.

Context:
{context}

Explain the topic: {topic}
"""
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()

# --- Study by Category (list subtopics and generate content) ---

def study_by_category(category):
    prompt = f"""
You are an aviation tutor for Canadian PPL students. Provide a clear, concise summary of the key subtopics in the category "{category}". 
Then give a brief explanation of each subtopic.

Format:
1. Subtopic name: explanation

Begin:
"""
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()

# --- Streamlit UI ---

st.title("🇨🇦 PPL Aviation Tutor (Transport Canada)")

mode = st.sidebar.selectbox("Select Study Mode", [
    "🔎 Ask a Question",
    "🧠 Quiz Me",
    "🧾 Explain a Topic",
    "📚 Study by Category"
])

if mode == "🔎 Ask a Question":
    st.write("Ask questions about aviation theory and get clear, simple explanations based on Canadian documents.")
    question = st.text_input("✈️ Ask a question about aviation...")
    if question:
        with st.spinner("Thinking like a flight instructor..."):
            answer = ask_tutor(question)
        st.markdown("### 🧠 Answer")
        st.write(answer)

elif mode == "🧠 Quiz Me":
    st.write("Get a multiple choice quiz on a topic. Select your answer and get immediate feedback.")
    topic = st.text_input("Enter a quiz topic (e.g., Air Law, Weather, Navigation)")
    num_questions = st.slider("Number of questions", 1, 10, 5)
    if st.button("Generate Quiz"):
        with st.spinner("Generating quiz questions..."):
            raw_quiz = generate_quiz_questions(topic, num_questions)
            quiz = parse_quiz(raw_quiz)
            if quiz:
                st.session_state['quiz'] = quiz
                st.session_state['current_q'] = 0
                st.session_state['score'] = 0
            else:
                st.warning("Could not generate quiz. Try a different topic.")
    if 'quiz' in st.session_state and st.session_state['quiz']:
        quiz = st.session_state['quiz']
        current_q = st.session_state['current_q']

        question_data = quiz[current_q]
        st.markdown(f"**Question {current_q+1} of {len(quiz)}:**")
        st.write(question_data['question'])
        
        user_answer = st.radio("Select your answer:", 
                               options=["A", "B", "C", "D"], 
                               format_func=lambda x: f"{x}: {question_data['choices'][x]}")

        if st.button("Submit Answer"):
            correct = question_data['correct_answer']
            if user_answer == correct:
                st.success("✅ Correct!")
                st.session_state['score'] += 1
            else:
                st.error(f"❌ Incorrect. Correct answer: {correct}: {question_data['choices'][correct]}")

            if current_q + 1 < len(quiz):
                st.session_state['current_q'] += 1
            else:
                st.markdown(f"### Quiz complete! Your score: {st.session_state['score']} / {len(quiz)}")
                # Clear quiz from session state to allow new quiz generation
                del st.session_state['quiz']
                del st.session_state['current_q']
                del st.session_state['score']

elif mode == "🧾 Explain a Topic":
    st.write("Enter a topic and get a simple explanation based on Canadian aviation documents.")
    topic = st.text_input("Enter a topic to explain...")
    if topic:
        with st.spinner("Generating explanation..."):
            explanation = explain_topic(topic)
        st.markdown("### Explanation")
        st.write(explanation)

elif mode == "📚 Study by Category":
    st.write("Select an aviation category to study. Get key subtopics and brief explanations.")
    categories = ["Air Law", "Weather", "Navigation", "Aircraft Performance", "Human Factors", "Meteorology", "Flight Instruments"]
    category = st.selectbox("Select category", categories)
    if st.button("Show Study Content"):
        with st.spinner("Fetching study content..."):
            study_content = study_by_category(category)
        st.markdown(f"### Study: {category}")
        st.write(study_content)
