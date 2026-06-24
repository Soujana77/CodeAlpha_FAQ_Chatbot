import streamlit as st
import time
from streamlit_mic_recorder import mic_recorder
from chatbot import (
    initialize_chatbot,
    get_answer,
    get_top_matches
)

def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

st.set_page_config(
    page_title="AI FAQ Assistant",
    page_icon="🤖",
    layout="wide"
)

# ---------- CUSTOM CSS ----------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* Main background */
.stApp {
    background: #0f172a;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #1f2937;
}

section[data-testid="stSidebar"] * {
    color: white !important;
    font-weight: 600 !important;
}

/* Hero */
.hero {
    text-align: center;
    padding: 40px;
    border-radius: 24px;
    background: linear-gradient(
        135deg,
        #2563eb,
        #7c3aed
    );
    color: white;
    margin-bottom: 30px;
    box-shadow: 0px 15px 40px rgba(37,99,235,0.3);
}

.hero h1 {
    font-size: 56px !important;
    font-weight: 800 !important;
    margin-bottom: 10px;
}

.hero p {
    font-size: 20px !important;
    font-weight: 500 !important;
}

/* Input box */
.stTextInput input {
    background: #1e293b !important;
    color: white !important;
    border: 2px solid #334155 !important;
    border-radius: 12px !important;
    padding: 14px !important;
    font-size: 18px !important;
    font-weight: 500 !important;
}

/* Ask button */
.stButton button {
    background: linear-gradient(
        135deg,
        #2563eb,
        #7c3aed
    ) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 24px !important;
    font-size: 16px !important;
    font-weight: 700 !important;
}

.stButton button:hover {
    transform: scale(1.03);
}

/* User message */
.user-bubble {
    padding: 18px;
    border-radius: 18px;
    background: linear-gradient(
        135deg,
        #2563eb,
        #3b82f6
    );
    color: white;
    font-size: 17px;
    font-weight: 600;
    margin-bottom: 12px;
}

/* Bot message */
.bot-bubble {
    padding: 18px;
    border-radius: 18px;
    background: #1e293b;
    color: white;
    font-size: 17px;
    font-weight: 500;
    margin-bottom: 12px;
    border: 1px solid #334155;
}

/* Headings */
h1 {
    font-weight: 800 !important;
}

h2 {
    font-size: 40px !important;
    font-weight: 800 !important;
    color: white !important;
}

h3 {
    font-weight: 700 !important;
    color: white !important;
}

/* Metrics */
[data-testid="stMetricValue"] {
    font-size: 32px !important;
    font-weight: 800 !important;
}

/* Footer */
.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 15px;
    margin-top: 40px;
}

</style>
""", unsafe_allow_html=True)

# ---------- CACHE ----------

@st.cache_resource
def load_chatbot():
    return initialize_chatbot()

faq_df, vectorizer, tfidf_matrix = load_chatbot()

# ---------- SESSION ----------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------- SIDEBAR ----------

with st.sidebar:
    st.title("🤖 AI FAQ Assistant")
    st.markdown("---")
    st.subheader("About Project")
    st.write("""
    AI-powered FAQ chatbot built using:
    • Python
    • NLP
    • TF-IDF
    • Cosine Similarity
    • Streamlit
    """)
    st.markdown("---")
    st.metric("Knowledge Base", f"{len(faq_df)} FAQs")
    st.metric("Conversations", len(st.session_state.chat_history))
    st.metric(
    "Categories",
    15
)
    
    if st.button("🗑 Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# ---------- HERO ----------

st.markdown("""
<div class="hero">
<h1>🤖 AI FAQ Assistant</h1>
<p>Ask questions about AI, Programming, Python, Machine Learning, NLP and Data Science</p>
</div>
""", unsafe_allow_html=True)

# ---------- WELCOME ----------

if len(st.session_state.chat_history) == 0:
    st.info("""
Try asking:
• What is Machine Learning?
• What is TensorFlow?
• Explain NLP
• What is GitHub?
• What is Cloud Computing?
""")

# ---------- INPUT ----------

st.subheader("💬 Ask a Question")

user_question = st.text_input("Type your question here...")

voice_data = mic_recorder(
    start_prompt="🎤 Start Recording",
    stop_prompt="⏹ Stop Recording",
    key="recorder"
)

if voice_data:
    st.success("🎙 Voice recorded successfully!")

if st.button("Ask"):
    if user_question.strip():
        with st.spinner("🤖 AI is thinking..."):
            time.sleep(1)
            answer, score = get_answer(
                user_question,
                faq_df,
                vectorizer,
                tfidf_matrix
            )
            top_matches = get_top_matches(
                user_question,
                faq_df,
                vectorizer,
                tfidf_matrix
            )

        confidence = round(score, 2)
        
        if confidence >= 0.80:
            label = "🟢 High Confidence"
        elif confidence >= 0.50:
            label = "🟡 Medium Confidence"
        else:
            label = "🔴 Low Confidence"

        st.session_state.chat_history.append({
            "question": user_question,
            "answer": answer,
            "confidence": confidence,
            "label": label,
            "matches": top_matches
        })

# ---------- CHAT ----------

st.markdown("""<h2>💬 Conversation</h2>""", unsafe_allow_html=True)

for chat in reversed(st.session_state.chat_history):
    st.markdown(f"""
    <div class="user-bubble">
    <b>👤 You</b><br>
    {chat['question']}
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="bot-bubble">
    <b>🤖 Assistant</b><br>
    {chat['answer']}
    </div>
    """, unsafe_allow_html=True)

    st.caption(f"{chat['label']} | Confidence Score: {chat['confidence']}")

    with st.expander("🔍 View Top Matches"):
        for i, match in enumerate(chat["matches"], start=1):
            st.write(f"{i}. {match['question']} ({match['score']})")

# ---------- FOOTER ----------

st.markdown("""
<div class="footer">
Built using Python • NLP • TF-IDF • Cosine Similarity • Streamlit
</div>
""", unsafe_allow_html=True)
