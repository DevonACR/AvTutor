import streamlit as st
st.set_page_config(page_title="Aviation Tutor 🇨🇦", layout="centered")

import json
import random
import requests
import os
import base64
import time  
import re
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
gemini_model = GenerativeModel(model_name="gemini-2.5-flash-lite")

@st.cache_data
def load_and_vectorize_chunks():
    """Load chunks and pre-compute TF-IDF vectorization (runs once per app restart)."""
    
    # Load the chunks
    with open("tc_chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    # Extract text and metadata
    chunk_texts = [chunk['content'] for chunk in chunks]
    chunk_sources = [chunk.get('source', 'Unknown') for chunk in chunks]
    
    # Pre-compute TF-IDF matrix (this is the expensive operation)
    vectorizer = TfidfVectorizer(
        max_features=10000,     # Limit vocabulary size for memory efficiency
        stop_words='english',   # Remove common words
        ngram_range=(1, 2),     # Include single words and pairs
        min_df=2,               # Ignore very rare terms
        max_df=0.95             # Ignore very common terms
    )
    
    # This expensive operation happens ONCE when app starts
    tfidf_matrix = vectorizer.fit_transform(chunk_texts)
    
    return chunks, chunk_texts, chunk_sources, vectorizer, tfidf_matrix

# Load everything once at startup
chunks, chunk_texts, chunk_sources, vectorizer, tfidf_matrix = load_and_vectorize_chunks()

@st.cache_data
def load_study_data():
    try:
        with open("ppl_study_topics_enriched.json", "r", encoding="utf-8") as f:
            study_topics = json.load(f)
    except FileNotFoundError:
        st.warning("Study Plan data not found. Using empty fallback.")
        study_topics = []

    # Load CARS index (changed from cars_data to cars_index to match your new structure)
    try:
        with open("cars_index.json", "r", encoding="utf-8") as f:
            cars_index = json.load(f)
    except FileNotFoundError:
        st.warning("CARS index not found. Using empty fallback.")
        cars_index = {}

    return study_topics, cars_index

study_topics, cars_index = load_study_data()

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

def search_chunks_fast(query: str, k: int = 4) -> List[Dict]:
    """Lightning-fast search using pre-computed vectors."""
    
    # Transform query using the EXISTING vectorizer (fast!)
    query_vec = vectorizer.transform([query])
    
    # Compute similarities (fast!)
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Get top results
    top_indices = similarities.argsort()[-k:][::-1]
    
    # Return results with similarity scores
    results = []
    for i in top_indices:
        results.append({
            "content": chunk_texts[i],
            "source": chunk_sources[i],
            "similarity_score": float(similarities[i])
        })
    
    return results

def get_categories():
    return sorted(set(chunk.get("category", "General") for chunk in chunks))

def ask_tutor_optimized(question: str, k: int = 2) -> str:
    start_time = time.time()

    # Fast search for context
    search_start = time.time()
    top_chunks = search_chunks_fast(question, k=k)
    search_time = time.time() - search_start

    # Prepare context (truncate for speed)
    prep_start = time.time()
    context_parts = []
    for chunk in top_chunks:
        content = chunk["content"]
        if len(content) > 250:
            content = content[:250] + "..."
        context_parts.append(content)
    context = "\n\n".join(context_parts)
    sources = list(set([chunk['source'] for chunk in top_chunks]))
    prep_time = time.time() - prep_start

    # GPT prompt
    prompt = f"""You are a Canadian PPL aviation tutor. Give clear, practical explanations.

Context:
{context}

Question: {question}

Provide a concise but complete answer. End with:
Study Source(s): {', '.join(sources)}"""

    # API call
    api_start = time.time()
    try:
        response = gemini_model.generate_content(prompt)
        result = response.text.strip()
    except Exception as e:
        result = f"⚠️ Gemini Error: {e}"
    api_time = time.time() - api_start
    total_time = time.time() - start_time

    # Performance metrics
    st.write("## ⏱️ Performance Breakdown")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🔍 Search", f"{search_time:.2f}s")
    with col2:
        st.metric("📝 Prep", f"{prep_time:.2f}s")
    with col3:
        st.metric("🤖 API", f"{api_time:.2f}s")
    with col4:
        st.metric("⏰ Total", f"{total_time:.2f}s")

    if total_time < 2:
        st.success("⚡ Excellent performance! Under 2 seconds.")
    elif total_time < 3:
        st.info("✅ Good performance! Under 3 seconds.")
    else:
        st.warning("🐌 Still room for improvement.")

    return result

def ask_tutor_expanded(original_question, original_answer):
    """Generate an expanded, detailed explanation based on the original Q&A."""
    start_time = time.time()
    
    # Get MORE chunks for detailed context (5 instead of 2)
    search_start = time.time()
    top_chunks = search_chunks_fast(original_question, k=5)
    search_time = time.time() - search_start
    
    # Use FULL content for expanded answers (no truncation)
    prep_start = time.time()
    detailed_context = "\n\n".join([chunk["content"] for chunk in top_chunks])
    sources = list(set([chunk['source'] for chunk in top_chunks]))
    
    # Enhanced prompt for detailed explanation
    prompt = f"""You are a Canadian PPL aviation tutor providing an EXPANDED, detailed explanation.

Original Question: {original_question}

Original Answer: {original_answer}

Detailed Context:
{detailed_context}

Provide a comprehensive, detailed explanation that:
1. Expands on the original answer with more depth
2. Includes practical examples and scenarios
3. Covers related concepts and connections
4. Explains the "why" behind regulations/procedures  
5. Adds safety considerations and real-world applications
6. Uses specific examples from Canadian aviation

Make this a thorough learning resource while keeping it clear and well-organized.

Study Source(s): {', '.join(sources)}"""
    
    prep_time = time.time() - prep_start
    
    # API call with detailed context
    api_start = time.time()
    try:
        response = gemini_model.generate_content(prompt)
        result = response.text.strip()
    except Exception as e:
        result = f"⚠️ Gemini Error: {e}"
    
    api_time = time.time() - api_start
    total_time = time.time() - start_time
    
    # Show performance for expanded answer
    st.write("### ⏱️ Expanded Answer Performance")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🔍 Search", f"{search_time:.2f}s")
    with col2:
        st.metric("📝 Prep", f"{prep_time:.2f}s")
    with col3:
        st.metric("🤖 API", f"{api_time:.2f}s")
    with col4:
        st.metric("⏰ Total", f"{total_time:.2f}s")
    
    if total_time < 8:
        st.info("📚 Detailed explanation generated successfully!")
    else:
        st.warning("🐌 Expanded answers take longer due to more comprehensive context.")
    
    return result

def study_plan_ui():
    st.subheader("📘 Study Plan Guide")

    if "study_progress" not in st.session_state:
        st.session_state.study_progress = {}

    # Check if study_topics is empty or not a list
    if not study_topics or not isinstance(study_topics, list):
        st.error("⚠️ Study topics data is not properly loaded or is empty.")
        return

    # Step 1: Categories
    categories = sorted(set(item["category"] for item in study_topics))
    selected_category = st.selectbox("Choose Category", categories)

    # Step 2: Subcategories
    subcats = [item for item in study_topics if item["category"] == selected_category]
    available_subcats = sorted(set(x["subcategory"] for x in subcats))
    selected_subcat = st.selectbox("Choose Subcategory", available_subcats)

    # Step 3: Sections
    sections = [item for item in subcats if item["subcategory"] == selected_subcat]
    available_sections = sorted(set(x["section"] for x in sections))
    selected_section = st.selectbox("Choose Section", available_sections)

    # Step 4: Find the specific section entry and extract topics
    try:
        # Find the exact section entry that matches our selection
        section_entry = next(
            s for s in sections 
            if s["section"] == selected_section and s["subcategory"] == selected_subcat
        )
        
        # Extract the nested topics list
        topics_list = section_entry.get("topics", [])
        
        if not topics_list:
            st.warning("⚠️ No topics found for this section.")
            return
        
        # Step 5: Topics dropdown
        topic_options = [t["topic"] for t in topics_list]
        selected_topic = st.selectbox("Choose Topic", topic_options)
        
        # Find the selected topic entry
        topic_entry = next(t for t in topics_list if t["topic"] == selected_topic)
        
    except StopIteration:
        st.error("⚠️ Could not find matching section entry.")
        return
    except KeyError as e:
        st.error(f"⚠️ Missing expected key in data structure: {e}")
        return

    # --- FIXED INDENTATION BELOW ---
    # 1. Set a unique key for each topic
    topic_key = f"{selected_category} > {selected_subcat} > {selected_section} > {selected_topic}"

    # 2. Get the current value from study_progress (default False)
    current_val = st.session_state.study_progress.get(topic_key, False)

    # 3. Draw the checkbox, using only the value from study_progress!
    checked = st.checkbox(
        "✅ Mark as Studied",
        value=current_val,
        key=f"studied_{hash(topic_key)}"
    )

    # 4. Only update study_progress if the value changed
    if checked != current_val:
        st.session_state.study_progress[topic_key] = checked

    # Progress bar
    total_topics = sum(len(item.get("topics", [])) for item in study_topics)
    studied_count = sum(1 for v in st.session_state.study_progress.values() if v)
    st.progress(studied_count / total_topics if total_topics > 0 else 0, 
                text=f"Progress: {studied_count}/{total_topics} topics studied")

    # Display reference content using cars_index
    references = topic_entry.get("references", [])
    if references:
        st.subheader("📚 Study References")
        for ref in references:
            if ref in cars_index:
                st.subheader(f"📖 CARS Reference: {ref}")
                
                # Display the content from cars_index
                cars_content = cars_index[ref]
                if isinstance(cars_content, dict):
                    # If it's a structured object, display nicely
                    if 'title' in cars_content:
                        st.markdown(f"**{cars_content['title']}**")
                    if 'content' in cars_content:
                        st.write(cars_content['content'])
                    elif 'text' in cars_content:
                        st.write(cars_content['text'])
                    else:
                        # Display all key-value pairs
                        for key, value in cars_content.items():
                            if key != 'title':
                                st.write(f"**{key.title()}:** {value}")
                elif isinstance(cars_content, str):
                    st.write(cars_content)
                else:
                    st.write(str(cars_content))
                
                st.markdown("---")
            else:
                st.info(f"📘 Reference: CARS {ref} (content not yet available)")
    else:
        st.info("📘 No specific CARS references listed for this topic. Use the AI Tutor to explore related concepts.")
    
    # Optional: Show topic hierarchy for clarity
    with st.expander("🗂️ Current Topic Path"):
        st.write(f"**Category:** {selected_category}")
        st.write(f"**Subcategory:** {selected_subcat}")
        st.write(f"**Section:** {selected_section}")
        st.write(f"**Topic:** {selected_topic}")
        if references:
            st.write(f"**References:** {', '.join(references)}")

# UI
mode = st.sidebar.radio(
    "Choose Study Mode",
    ["💬 AI Tutor", "🧠 Quiz Me", "📚 Study by Category", "🧪 PPL Sample Exams", "🧩 Flashcards", "📘 Study Plan Guide"]
)

# ---------------- AI Tutor with Persistent Q&A ----------------
if mode == "💬 AI Tutor":
    st.subheader("💬 Ask or Learn Any Topic")

    # Initialize session state
    st.session_state.setdefault("tutor_input", "")
    st.session_state.setdefault("tutor_answer", "")
    st.session_state.setdefault("simplified_answer", "")
    st.session_state.setdefault("expanded_answer", "")

    # Temporary input field
    user_query = st.text_input(
        "✈️ Ask a question or enter a topic to learn:",
        value=st.session_state["tutor_input"],
        key="tutor_temp"
    )

    # Submit logic
    def submit_tutor_question():
        query = st.session_state.get("tutor_temp", "").strip()
        if query:
            # Clear previous answers to avoid showing old expanded content
            for k in ["tutor_answer", "simplified_answer", "expanded_answer"]:
                st.session_state.pop(k, None)

            st.session_state["tutor_input"] = query
            with st.spinner("Explaining like a ground school instructor..."):
                st.session_state["tutor_answer"] = ask_tutor_optimized(query)

    # Auto-submit when input changes
    if user_query and user_query != st.session_state["tutor_input"]:
        submit_tutor_question()

    # Manual submit button
    if st.button("🧠 Submit"):
        submit_tutor_question()

    # Show tutor answer
    if st.session_state["tutor_answer"]:
        st.markdown("### 🧠 Instructor Explanation")
        st.write(st.session_state["tutor_answer"])

    # Three-button layout: simplify, expand, clear
    col1, col2, col3 = st.columns(3)

    with col1:
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

    with col2:
        if st.button("📖 Expand on this topic"):
            with st.spinner("Generating detailed explanation..."):
                try:
                    expanded = ask_tutor_expanded(
                        st.session_state["tutor_input"], 
                        st.session_state["tutor_answer"]
                    )
                    st.session_state["expanded_answer"] = expanded
                except Exception as e:
                    st.session_state["expanded_answer"] = f"⚠️ Error expanding: {e}"

    with col3:
        if st.button("🗑 Clear all answers"):
            for k in ["tutor_input", "tutor_answer", "simplified_answer", "expanded_answer", "tutor_temp"]:
                st.session_state.pop(k, None)
            st.rerun()

    # Show simplified/expanded if available
    if st.session_state.get("simplified_answer"):
        st.markdown("### 🍼 Simplified Explanation")
        st.write(st.session_state["simplified_answer"])

    if st.session_state.get("expanded_answer"):
        st.markdown("### 📖 Detailed Deep-Dive")
        st.info("This expanded explanation uses more context and may take longer to generate.")
        st.write(st.session_state["expanded_answer"])

    # Clear button logic
    if not st.session_state.get("simplified_answer") and not st.session_state.get("expanded_answer"):
        if st.session_state.get("tutor_answer"):  # Only show clear button if there's an answer
            if st.button("🗑 Clear"):
                for k in ["tutor_input", "tutor_answer", "simplified_answer", "expanded_answer", "tutor_temp"]:
                    st.session_state.pop(k, None)
                st.rerun()
    
    # Show help text only when appropriate
    if not st.session_state.get("tutor_answer"):
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

    # Show results if user has submitted all questions
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

    # Continue with question display logic
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
        if st.button("Next ➡️", disabled=(current_q == len(quiz) - 1 or current_q not in st.session_state.quiz_submitted)):
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

    # Final result if all answers have been submitted
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

    # Session state initialization
    if "flashcards" not in st.session_state:
        st.session_state.flashcards = load_flashcards()
    if "user_flashcards" not in st.session_state:
        st.session_state.user_flashcards = []
    if "shuffled_flashcards" not in st.session_state:
        combined = st.session_state.flashcards + st.session_state.user_flashcards
        random.shuffle(combined)
        st.session_state.shuffled_flashcards = combined
    if "flash_index" not in st.session_state:
        st.session_state.flash_index = 0
    if "show_answer" not in st.session_state:
        st.session_state.show_answer = False
    if "known_cards" not in st.session_state:
        st.session_state.known_cards = set()

    all_flashcards = st.session_state.shuffled_flashcards

    # Topic filter
    available_topics = sorted(set(card.get("topic", "Unknown") for card in all_flashcards))
    selected_topic = st.selectbox("📘 Filter by Topic", ["All"] + available_topics)

    filtered_cards = [
        card for i, card in enumerate(all_flashcards)
        if (selected_topic == "All" or card.get("topic") == selected_topic)
        and i not in st.session_state.known_cards
    ]

    # Progress bar (based on all_flashcards, not filtered stack)
    total = len(all_flashcards)
    known = len(st.session_state.known_cards)
    st.progress(known / total if total else 0, text=f"{known}/{total} cards marked as known")

    if not filtered_cards:
        st.success("🎉 You've marked all cards in this topic as known!")
        if st.button("🔁 Reset All Known Cards"):
            st.session_state.known_cards = set()
            st.session_state.flash_index = 0
            st.session_state.show_answer = False
            st.rerun()
        st.stop()

    idx = st.session_state.flash_index
    idx = max(0, min(idx, len(filtered_cards) - 1))
    card = filtered_cards[idx]

    st.markdown(f"**Card {idx + 1} of {len(filtered_cards)}**")
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
        if st.button("➡️ Next") and idx < len(filtered_cards) - 1:
            st.session_state.flash_index += 1
            st.session_state.show_answer = False
            st.rerun()
    with col4:
        if st.button("✅ I Know This"):
            global_index = all_flashcards.index(card)
            st.session_state.known_cards.add(global_index)
            st.session_state.flash_index = min(idx, len(filtered_cards) - 2)
            st.session_state.show_answer = False
            st.rerun()

    st.markdown("___")
    if st.button("🔁 Reset All Known Cards"):
        st.session_state.known_cards = set()
        st.session_state.flash_index = 0
        st.session_state.show_answer = False
        st.rerun()

    # Flashcard creation form
    with st.expander("➕ Create Your Own Flashcard"):
        q = st.text_input("📝 Question")
        a = st.text_area("💡 Answer")
        topic_input = st.text_input("📘 Topic (optional)", value="User Generated")
        if st.button("➕ Add Flashcard") and q.strip() and a.strip():
            new_card = {
                "question": q.strip(),
                "answer": a.strip(),
                "topic": topic_input.strip() or "User Generated",
                "source": "Custom"
            }
            st.session_state.user_flashcards.append(new_card)
            # Regenerate shuffled list
            combined = st.session_state.flashcards + st.session_state.user_flashcards
            random.shuffle(combined)
            st.session_state.shuffled_flashcards = combined
            st.success("✅ Flashcard added!")
            st.rerun()

elif mode == "📘 Study Plan Guide":
    study_plan_ui()









