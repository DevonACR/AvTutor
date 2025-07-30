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

@st.cache_data
def load_chunks():
    with open("tc_chunks.json", "r", encoding="utf-8") as f:
        return json.load(f)

chunks = load_chunks()
chunk_texts = [chunk['content'] for chunk in chunks]
chunk_sources = [chunk.get('source', 'Unknown') for chunk in chunks]

@st.cache_data
def load_sample_exam_questions():
    url = "https://raw.githubusercontent.com/DevonACR/AvTutor/main/sample_exam_structured.json"
    return requests.get(url).json()

@st.cache_data
def load_flashcards():
    url = "https://raw.githubusercontent.com/DevonACR/AvTutor/main/generated_flashcards.json"
    res = requests.get(url)
    return res.json() if res.status_code == 200 else []

@st.cache_data
def load_generated_quiz_questions():
    url = "https://raw.githubusercontent.com/DevonACR/AvTutor/main/generated_quiz_questions.json"
    res = requests.get(url)
    return res.json() if res.status_code == 200 else []

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
    prompt = f"""You are a Canadian PPL aviation tutor. Explain concepts clearly and simply.

Context:
{context}

Question: {question}

Answer with explanation, then end with:

Study Source(s): {', '.join(sources)}"""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Gemini Error: {e}"


def get_categories():
    return sorted(set(chunk.get("category", "General") for chunk in chunks))

def get_questions_by_category(category: str, limit: int = 25) -> List[Dict]:
    filtered = chunks if category == "All" else [c for c in chunks if c.get("category") == category]
    return random.sample(filtered, min(len(filtered), limit))

# UI
mode = st.sidebar.radio(
    "Choose Study Mode",
    ["💬 AI Tutor", "🧠 Quiz Me", "📚 Study by Category", "🧪 PPL Sample Exams", "🧩 Flashcards"]
)

# ---------------- AI Tutor with Persistent Q&A ----------------
if mode == "💬 AI Tutor":
    st.subheader("💬 Ask or Learn Any Topic")

    # Initialize session state variables
    st.session_state.setdefault("tutor_input", "")
    st.session_state.setdefault("tutor_answer", "")
    st.session_state.setdefault("simplified_answer", "")

    # Temporary input field for syncing (does not auto-trigger a rerun)
    user_query = st.text_input(
        "✈️ Ask a question or enter a topic to learn:",
        value=st.session_state["tutor_input"],
        key="tutor_temp"
    )

    # Submit logic (called on button click or if Enter is pressed)
    def submit_tutor_question():
        query = st.session_state.get("tutor_temp", "").strip()
        if query:
            st.session_state["tutor_input"] = query
            with st.spinner("Explaining like a ground school instructor..."):
                st.session_state["tutor_answer"] = ask_tutor(query)
            st.session_state["simplified_answer"] = ""

    # Auto-submit when Enter is pressed (if input changed)
    if user_query and user_query != st.session_state["tutor_input"]:
        submit_tutor_question()

    # Submit button (manual trigger)
    if st.button("🧠 Submit"):
        submit_tutor_question()

    # Show response
    if st.session_state["tutor_answer"]:
        st.markdown("### 🧠 Instructor Explanation")
        st.write(st.session_state["tutor_answer"])

        if st.button("🍼 Simplify this explanation"):
            with st.spinner("Making it super beginner-friendly..."):
                try:
                    simplified_prompt = (
                        "You are an aviation tutor. Simplify this explanation for a beginner:\n\n"
                        f"{st.session_state['tutor_answer']}"
                    )
                    simplified = gemini_model.generate_content(simplified_prompt).text.strip()
                    st.session_state["simplified_answer"] = simplified
                except Exception as e:
                    st.session_state["simplified_answer"] = f"⚠️ Error simplifying: {e}"

        if st.session_state["simplified_answer"]:
            st.markdown("### 🍼 Simplified Explanation")
            st.write(st.session_state["simplified_answer"])

        if st.button("🗑 Clear"):
            for k in ["tutor_input", "tutor_answer", "simplified_answer", "tutor_temp"]:
                st.session_state.pop(k, None)
            st.rerun()
    else:
        st.info("Ask a question above to get an explanation. Use 'Simplify' for beginner-friendly wording.")



elif mode == "📚 Study by Category":
    st.subheader("📚 Study Notes by Category")
    categories = get_categories()
    selected_category = st.selectbox("Select a topic category:", categories)
    filtered = [chunk for chunk in chunks if chunk.get("category") == selected_category]
    for i, chunk in enumerate(filtered):
        with st.expander(f"📘 Note {i+1}"):
            st.write(chunk['content'])
            st.caption(f"📚 Source: {chunk.get('source', 'Unknown')}")

elif mode == "🧠 Quiz Me":
    st.subheader("🧠 Quiz Me")

    all_questions = load_generated_quiz_questions()
    categories = sorted(set(q.get("topic", "General") for q in all_questions))
    selected_category = st.selectbox("📚 Choose a category", ["All"] + categories)

    if (
        "quiz" not in st.session_state
        or st.session_state.get("quiz_category") != selected_category
    ):
        questions = [
            q for q in all_questions
            if selected_category == "All" or q.get("topic", "General") == selected_category
        ]

        if not questions:
            st.warning("⚠️ No quiz questions found for this category.")
            if st.button("🔁 Start Over"):
                for k in ["quiz", "quiz_category", "quiz_index", "quiz_answers", "quiz_submitted"]:
                    st.session_state.pop(k, None)
                st.rerun()
            st.stop()

        random.shuffle(questions)
        st.session_state.quiz = questions
        st.session_state.quiz_category = selected_category
        st.session_state.quiz_index = 0
        st.session_state.quiz_answers = {}
        st.session_state.quiz_submitted = set()

    quiz = st.session_state.quiz
    current_q = st.session_state.quiz_index

    # ✅ Show results if user has submitted all questions
    if len(st.session_state.quiz_submitted) == len(quiz):
        total = len(quiz)
        correct = 0
        for i, q in enumerate(quiz):
            ans = st.session_state.quiz_answers.get(i, "")
            answer_letter = q.get("answer")
            options = q.get("choices") or q.get("options", [])
            if isinstance(options, dict):
                options = [f"{k}: {v}" for k, v in options.items()]
            correct_opt = next((opt for opt in options if opt.startswith(answer_letter)), "")
            if ans == correct_opt:
                correct += 1

        score = correct / total * 100
        st.markdown("## ✅ Quiz Results")
        st.success(f"🎯 You scored {correct} / {total} ({score:.1f}%)")

        if st.button("🔁 Start Again"):
            for k in ["quiz", "quiz_category", "quiz_index", "quiz_answers", "quiz_submitted"]:
                st.session_state.pop(k, None)
            st.rerun()
        st.stop()

    # ✅ Continue with question display logic
    if current_q >= len(quiz):
        current_q = len(quiz) - 1
        st.session_state.quiz_index = current_q

    q_data = quiz[current_q]

    # Safely extract options
    raw_choices = q_data.get("choices") or q_data.get("options")
    if isinstance(raw_choices, dict):
        options = [f"{k}: {v}" for k, v in raw_choices.items()]
    elif isinstance(raw_choices, list):
        options = raw_choices
    else:
        st.error(f"⚠️ Malformed question format at index {current_q}. Skipping...")
        st.session_state.quiz_index += 1
        st.rerun()

    q_key = f"quiz_q_{current_q}"
    st.markdown(f"**Question {current_q + 1} of {len(quiz)}**")
    st.write(q_data["question"])

    prev_answer = st.session_state.quiz_answers.get(current_q)
    default_index = options.index(prev_answer) if prev_answer in options else 0

    user_selection = st.radio(
        "Select your answer:",
        options,
        index=default_index,
        key=q_key
    )

    if st.button("✅ Submit Answer"):
        st.session_state.quiz_answers[current_q] = user_selection
        st.session_state.quiz_submitted.add(current_q)

        if current_q + 1 >= len(quiz):
            st.session_state.quiz_index = len(quiz)  # trigger results
        else:
            st.session_state.quiz_index = current_q + 1

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
            st.session_state.quiz_index = max(0, current_q - 1)
            st.rerun()
    with col2:
        if st.button("Next ➡️", disabled=(current_q == len(quiz) - 1)):
            st.session_state.quiz_index = min(len(quiz) - 1, current_q + 1)
            st.rerun()


elif mode == "🧪 PPL Sample Exams":
    st.subheader("🧪 Official Sample Exam Practice")

    questions = load_sample_exam_questions()
    total_available = len(questions)
    num_questions = st.slider("How many questions would you like to attempt?", 1, min(100, total_available), 10)

    # Always initialize submitted answers tracking
    if "sample_exam_submitted" not in st.session_state:
        st.session_state.sample_exam_submitted = set()

    # Initialize or reset state
    if "sample_exam_set" not in st.session_state or st.session_state.get("sample_exam_len") != num_questions:
        st.session_state.sample_exam_set = random.sample(questions, num_questions)
        st.session_state.sample_exam_index = 0
        st.session_state.sample_exam_answers = {}
        st.session_state.sample_exam_len = num_questions
        st.session_state.sample_exam_submitted = set()

    q_index = st.session_state.sample_exam_index
    current_question = st.session_state.sample_exam_set[q_index]
    question_key = f"sample_q_{q_index}"

    st.markdown(f"**Question {q_index + 1} of {num_questions}**")
    st.markdown(current_question["question"])

    if "images" in current_question:
        for img in current_question["images"]:
            url = f"https://raw.githubusercontent.com/DevonACR/AvTutor/main/exam_visuals/{img}"
            st.image(url, use_container_width=True)
    elif "image" in current_question:
        url = f"https://raw.githubusercontent.com/DevonACR/AvTutor/main/exam_visuals/{current_question['image']}"
        st.image(url, use_container_width=True)

    previous_answer = st.session_state.sample_exam_answers.get(q_index, None)
    user_selection = st.radio(
        "Select your answer:",
        current_question["options"],
        index=current_question["options"].index(previous_answer)
        if previous_answer in current_question["options"] else 0,
        key=question_key
    )

    if st.button("✅ Submit Answer"):
        st.session_state.sample_exam_answers[q_index] = user_selection
        st.session_state.sample_exam_submitted.add(q_index)
        st.rerun()

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

    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Previous", disabled=(q_index == 0)):
            st.session_state.sample_exam_index -= 1
            st.rerun()
    with col2:
        if st.button("Next ➡️", disabled=(q_index == num_questions - 1)):
            st.session_state.sample_exam_index += 1
            st.rerun()


elif mode == "🧩 Flashcards":
    st.subheader("🧩 Flashcard Study Mode")

    # ✅ Load flashcards and user cards
    if "flashcards" not in st.session_state:
        st.session_state.flashcards = load_flashcards()
        st.session_state.user_flashcards = []
        st.session_state.flash_index = 0
        st.session_state.show_answer = False
        st.session_state.known_cards = set()

    # ✅ Combine default + user flashcards, filter known
    all_flashcards = st.session_state.flashcards + st.session_state.user_flashcards
    cards = [
        card for i, card in enumerate(all_flashcards)
        if i not in st.session_state.known_cards
    ]

    if not cards:
        st.success("🎉 You've marked all cards as known!")
        if st.button("🔁 Reset All Known Cards"):
            st.session_state.known_cards = set()
            st.session_state.flash_index = 0
            st.session_state.show_answer = False
            st.rerun()
        st.stop()

    idx = st.session_state.flash_index
    idx = max(0, min(idx, len(cards) - 1))  # prevent index error
    card = cards[idx]

    st.markdown(f"**Card {idx + 1} of {len(cards)}**")
    st.subheader(f"❓ {card['question']}")
    st.caption(f"📘 Topic: {card.get('topic', 'User Added' if card in st.session_state.user_flashcards else 'Unknown')}")

    if st.session_state.show_answer:
        st.success(f"✅ {card['answer']}")
        st.caption(f"📚 Source: {card.get('source', 'Unknown')}")

    col1, col2, col3, col4 = st.columns(4)
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
        if st.button("➡️ Next") and idx < len(cards) - 1:
            st.session_state.flash_index += 1
            st.session_state.show_answer = False
            st.rerun()
    with col4:
        if st.button("✅ I Know This"):
            global_index = all_flashcards.index(card)
            st.session_state.known_cards.add(global_index)
            st.session_state.flash_index = min(idx, len(cards) - 2)
            st.session_state.show_answer = False
            st.rerun()

    st.markdown("___")
    if st.button("🔁 Reset All Known Cards"):
        st.session_state.known_cards = set()
        st.session_state.flash_index = 0
        st.session_state.show_answer = False
        st.rerun()

    # ➕ Flashcard creation form
    with st.expander("➕ Create Your Own Flashcard"):
        q = st.text_input("📝 Question")
        a = st.text_area("💡 Answer")
        if st.button("➕ Add Flashcard") and q.strip() and a.strip():
            new_card = {
                "question": q.strip(),
                "answer": a.strip(),
                "topic": "User Generated",
                "source": "Custom"
            }
            st.session_state.user_flashcards.append(new_card)
            st.success("✅ Flashcard added!")
            st.rerun()
