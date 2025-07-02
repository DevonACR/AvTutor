import streamlit as st
st.set_page_config(page_title="Aviation Tutor 🇨🇦", layout="centered")

import json
import random
import requests
import os
import base64
from typing import List, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# ✅ DECODE GOOGLE CREDENTIALS
b64_key = st.secrets["GOOGLE_KEY_B64"]
decoded_key = base64.b64decode(b64_key)
with open("/tmp/gemini-key.json", "wb") as f:
    f.write(decoded_key)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/gemini-key.json"

# ✅ Import and initialize VertexAI
from vertexai.generative_models import GenerativeModel
import vertexai

vertexai.init(project="gen-lang-client-0636505424", location="us-central1")
gemini_model = GenerativeModel(model_name="gemini-2.5-pro")



# Load theory chunks from tc_chunks.json
@st.cache_data

def load_chunks():
    with open("tc_chunks.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

chunks = load_chunks()
chunk_texts = [chunk['content'] for chunk in chunks]
chunk_sources = [chunk.get('source', 'Unknown') for chunk in chunks]

# Load sample exam questions from GitHub
@st.cache_data

def load_sample_exam_questions():
    url = "https://raw.githubusercontent.com/DevonACR/AvTutor/main/sample_exam_structured.json"  # Replace with your username/repo
    res = requests.get(url)
    return res.json()

# 🔁 Load Flashcard Prompts from GitHub
@st.cache_data
def load_flashcards():
    url = "https://raw.githubusercontent.com/DevonACR/AvTutor/main/generated_flashcards.json"
    res = requests.get(url)

    if res.status_code != 200:
        st.error(f"❌ Failed to fetch flashcards - HTTP {res.status_code}")
        return []

    try:
        return res.json()
    except Exception as e:
        st.error(f"❌ Failed to parse flashcards: {e}")
        st.code(res.text[:500], language="json")
        return []


# TF-IDF for search
vectorizer = TfidfVectorizer().fit_transform(chunk_texts)

def search_chunks(query: str, k: int = 5) -> List[Dict]:
    query_vec = TfidfVectorizer().fit(chunk_texts).transform([query])
    sims = cosine_similarity(query_vec, vectorizer).flatten()
    top_indices = sims.argsort()[-k:][::-1]
    return [{"content": chunk_texts[i], "source": chunk_sources[i]} for i in top_indices]

def ask_tutor(question):
    top_chunks = search_chunks(question)
    context = "\n\n".join([chunk["content"] for chunk in top_chunks])
    sources = [chunk['source'] for chunk in top_chunks]

    prompt = f"""
You are a Canadian PPL aviation tutor. Explain concepts clearly and simply.

Context:
{context}

Question: {question}

Answer with explanation, then end with:

Study Source(s): {', '.join(sources)}
"""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Gemini Error: {e}"

def get_categories():
    cats = [chunk.get("category", "General") for chunk in chunks]
    return sorted(list(set(cats)))

def get_questions_by_category(category: str, limit: int = 25) -> List[Dict]:
    if category == "All":
        filtered = chunks
    else:
        filtered = [chunk for chunk in chunks if chunk.get("category") == category]
    sampled = random.sample(filtered, min(len(filtered), limit))
    return sampled

# Streamlit UI
mode = st.sidebar.radio(
    "Choose Study Mode",
    ["🔎 Ask a Question", "🧠 Quiz Me", "🧾 Explain a Topic", "📚 Study by Category", "🧪 PPL Sample Exams", "🧩 Flashcards"]
)

if mode == "🔎 Ask a Question":
    st.write("Ask questions about aviation theory and get clear, simple explanations based on Canadian documents.")
    question = st.text_input("✈️ Ask a question about aviation...")
    if question:
        with st.spinner("Thinking like a flight instructor..."):
            answer = ask_tutor(question)
        st.markdown("### 🧠 Answer")
        st.write(answer)

elif mode == "📚 Study by Category":
    st.subheader("📚 Study Notes by Category")
    categories = get_categories()
    selected_category = st.selectbox("Select a topic category:", categories)
    filtered = [chunk for chunk in chunks if chunk.get("category") == selected_category]
    for i, chunk in enumerate(filtered):
        with st.expander(f"📘 Note {i+1}"):
            st.write(chunk['content'])
            st.caption(f"📚 Source: {chunk.get('source', 'Unknown')}")

elif mode == "🧾 Explain a Topic":
    st.subheader("🧾 Explain a Topic")
    topic = st.text_input("What topic do you want explained?")
    if topic:
        with st.spinner("Explaining like a ground school instructor..."):
            answer = ask_tutor(f"Explain {topic} in simple terms.")
        st.markdown("### 🧠 Explanation")
        st.write(answer)

elif mode == "🧠 Quiz Me":
    st.subheader("🧠 Quiz Me")

    # Categories and question pool
    categories = sorted(set(chunk.get("category", "General") for chunk in chunks))
    selected_category = st.selectbox("📚 Choose a category", ["All"] + categories)

    if "quiz" not in st.session_state or st.session_state.get("quiz_category") != selected_category:
        st.session_state.quiz_category = selected_category
        st.session_state.quiz_index = 0
        st.session_state.quiz_answers = {}
        st.session_state.quiz_submitted = set()

        filtered_chunks = chunks if selected_category == "All" else [c for c in chunks if c.get("category") == selected_category]

        questions = []
        for chunk in filtered_chunks:
            if "quiz_question" in chunk:
                q = chunk["quiz_question"]
                questions.append({
                    "question": q["question"],
                    "options": [f"{k}: {v}" for k, v in q["choices"].items()],
                    "answer": q["answer"],
                    "source": chunk.get("source", "Unknown")
                })

        if not questions:
            st.warning("⚠️ No quiz questions found for this category.")
            st.stop()

        random.shuffle(questions)
        st.session_state.quiz = questions


    quiz = st.session_state.quiz

if not quiz:
    st.warning("⚠️ No quiz questions available for this category.")
    st.stop()

current_q = st.session_state.quiz_index

if current_q >= len(quiz):
    st.success("🎉 You've completed all questions in this category!")
    st.stop()


    q_data = quiz[current_q]
    q_key = f"quiz_q_{current_q}"

    st.markdown(f"**Question {current_q + 1}**")
    st.write(q_data["question"])

    prev_ans = st.session_state.quiz_answers.get(current_q)
    options = q_data["options"]
    default_index = options.index(prev_ans) if prev_ans in options else 0

    user_selection = st.radio(
        "Select your answer:",
        options,
        index=default_index,
        key=q_key
    )

    if st.button("✅ Submit Answer"):
        st.session_state.quiz_answers[current_q] = user_selection
        st.session_state.quiz_submitted.add(current_q)
        st.rerun()

    if current_q in st.session_state.quiz_submitted:
        correct_letter = q_data["answer"]
        correct_option = next((opt for opt in options if opt.startswith(correct_letter)), None)

        if user_selection == correct_option:
            st.success("✅ Correct!")
        else:
            st.error(f"❌ Incorrect. Correct answer: {correct_option}")

        st.caption(f"📘 Source: {q_data.get('source', 'Unknown')}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Previous", disabled=(current_q == 0)):
            st.session_state.quiz_index -= 1
            st.rerun()
    with col2:
        if st.button("Next ➡️", disabled=(current_q == len(quiz) - 1)):
            st.session_state.quiz_index += 1
            st.rerun()



elif mode == "🧪 PPL Sample Exams":
    st.subheader("🧪 Official Sample Exam Practice")

    questions = load_sample_exam_questions()
    total_available = len(questions)
    num_questions = st.slider("How many questions would you like to attempt?", 1, min(100, total_available), 10)

    # ✅ Always initialize submitted answers tracking
    if "sample_exam_submitted" not in st.session_state:
        st.session_state.sample_exam_submitted = set()


    # Initialize or reset state
    if "sample_exam_set" not in st.session_state or st.session_state.get("sample_exam_len") != num_questions:
        st.session_state.sample_exam_set = random.sample(questions, num_questions)
        st.session_state.sample_exam_index = 0
        st.session_state.sample_exam_answers = {}
        st.session_state.sample_exam_len = num_questions

    # ✅ Ensure this always exists, even if state isn’t reset
    if "sample_exam_submitted" not in st.session_state:
        st.session_state.sample_exam_submitted = set()



    q_index = st.session_state.sample_exam_index
    current_question = st.session_state.sample_exam_set[q_index]
    question_key = f"sample_q_{q_index}"

    st.markdown(f"**Question {q_index + 1} of {num_questions}**")
    st.markdown(current_question["question"])

    # Image display
    if "images" in current_question:
        for img in current_question["images"]:
            url = f"https://raw.githubusercontent.com/DevonACR/AvTutor/main/exam_visuals/{img}"
            st.image(url, use_container_width=True)
    elif "image" in current_question:
        url = f"https://raw.githubusercontent.com/DevonACR/AvTutor/main/exam_visuals/{current_question['image']}"
        st.image(url, use_container_width=True)

    # Radio with persistence
    previous_answer = st.session_state.sample_exam_answers.get(q_index, None)
    user_selection = st.radio(
        "Select your answer:",
        current_question["options"],
        index=current_question["options"].index(previous_answer)
        if previous_answer in current_question["options"] else 0,
        key=question_key
    )

    # Submit logic with rerun
    if st.button("✅ Submit Answer"):
        st.session_state.sample_exam_answers[q_index] = user_selection
        st.session_state.sample_exam_submitted.add(q_index)
        st.rerun()

    # Feedback
    if q_index in st.session_state.sample_exam_submitted:
        correct_letter = current_question["answer"]
        correct_option = [opt for opt in current_question["options"] if opt.startswith(correct_letter)][0]

        if st.session_state.sample_exam_answers.get(q_index) == correct_option:
            st.success("✅ Correct!")
        else:
            st.error(f"❌ Incorrect. Correct answer: {correct_option}")

        if "references" in current_question:
            for ref in current_question["references"]:
                st.caption(f"📘 Reference: {ref}")
        elif "reference" in current_question:
            st.caption(f"📘 Reference: {current_question['reference']}")

    # Navigation buttons (with rerun)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Previous", disabled=(q_index == 0)):
            st.session_state.sample_exam_index -= 1
            st.rerun()
    with col2:
        if st.button("Next ➡️", disabled=(q_index == num_questions - 1)):
            st.session_state.sample_exam_index += 1
            st.rerun()

    # ✅ Final result if all answers have been submitted
if len(st.session_state.sample_exam_submitted) == num_questions:
    correct_total = 0
    incorrect = []

    for i, q in enumerate(st.session_state.sample_exam_set):
        user_ans = st.session_state.sample_exam_answers.get(i, "")
        correct_opt = [opt for opt in q["options"] if opt.startswith(q["answer"])]
        if correct_opt and user_ans == correct_opt[0]:
            correct_total += 1
        else:
            incorrect.append((i, q, user_ans, correct_opt[0] if correct_opt else "Unknown"))

    score = correct_total / num_questions * 100
    passed = score >= 70

    st.markdown("## 📝 Exam Results")
    st.success(f"🎯 Your Score: {correct_total} / {num_questions} ({score:.1f}%)")
    if passed:
        st.balloons()
        st.success("✅ You passed the sample exam! (70%+)")
    else:
        st.error("❌ You did not pass. Review the references and try again.")

    # 🔍 Show incorrect answers with references
    if incorrect:
        st.markdown("### 🔍 Review Incorrect Answers")
        for i, q, user_ans, correct_ans in incorrect:
            st.markdown(f"**Question {i+1}:** {q['question']}")
            st.error(f"❌ Your Answer: {user_ans}")
            st.success(f"✅ Correct Answer: {correct_ans}")
            if "references" in q:
                for ref in q["references"]:
                    st.caption(f"📘 Reference: {ref}")
            elif "reference" in q and q["reference"]:
                st.caption(f"📘 Reference: {q['reference']}")
            st.markdown("---")





elif mode == "🧩 Flashcards":
    st.subheader("🧩 Flashcard Study Mode")

    if "flashcards" not in st.session_state:
        st.session_state.flashcards = load_flashcards()
        st.session_state.flash_index = 0
        st.session_state.show_answer = False

    cards = st.session_state.flashcards
    idx = st.session_state.flash_index
    card = cards[idx]

    st.markdown(f"**Card {idx + 1} of {len(cards)}**")
    st.subheader(f"❓ {card['question']}")
    st.caption(f"📘 Topic: {card.get('topic', 'Unknown')}")

    # Show or hide answer
    if st.session_state.show_answer:
        st.success(f"✅ {card['answer']}")
        st.caption(f"📚 Source: {card.get('source', 'Unknown')}")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("⬅️ Previous") and idx > 0:
            st.session_state.flash_index -= 1
            st.session_state.show_answer = False
            st.rerun()

    with col2:
        if st.button("🔄 Show Answer" if not st.session_state.show_answer else "🔁 Hide Answer"):
            st.session_state.show_answer = not st.session_state.show_answer
            st.rerun()

    with col3:
        if st.button("Next ➡️") and idx < len(cards) - 1:
            st.session_state.flash_index += 1
            st.session_state.show_answer = False
            st.rerun()
