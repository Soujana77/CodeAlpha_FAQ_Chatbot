import streamlit as st

from chatbot import (
    initialize_chatbot,
    get_answer,
    get_top_matches
)

st.set_page_config(
    page_title="AI FAQ Assistant",
    page_icon="🤖",
    layout="wide"
)

# ---------- CUSTOM CSS ----------

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.hero {
    text-align: center;
    padding: 1.5rem;
    border-radius: 15px;
    background: linear-gradient(90deg, #1f2937, #374151);
    color: white;
    margin-bottom: 2rem;
}

.user-bubble {
    padding: 15px;
    border-radius: 15px;
    background-color: #2563eb;
    color: white;
    margin-bottom: 10px;
}

.bot-bubble {
    padding: 15px;
    border-radius: 15px;
    background-color: #f3f4f6;
    color: black;
    margin-bottom: 10px;
}

.metric-card {
    padding: 1rem;
    border-radius: 12px;
    background-color: #f3f4f6;
    text-align: center;
}

.footer {
    text-align: center;
    color: gray;
    margin-top: 30px;
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

    st.metric(
        "Knowledge Base",
        f"{len(faq_df)} FAQs"
    )

    st.metric(
        "Conversations",
        len(st.session_state.chat_history)
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

user_question = st.text_input(
    "Ask your question:"
)

if st.button("Ask"):

    if user_question.strip():

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

        st.session_state.chat_history.append(
            {
    "question": user_question,
    "answer": answer,
    "confidence": confidence,
    "label": label,
    "matches": top_matches
}
        )

# ---------- CHAT ----------

st.markdown("## 💬 Conversation")

for chat in reversed(st.session_state.chat_history):

    st.markdown(
        f"""
<div class="user-bubble">
<b>👤 You</b><br>
{chat['question']}
</div>
""",
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
<div class="bot-bubble">
<b>🤖 Assistant</b><br>
{chat['answer']}
</div>
""",
        unsafe_allow_html=True
    )

    st.caption(
        f"{chat['label']} | Confidence Score: {chat['confidence']}"
    )
with st.expander("🔍 View Top Matches"):

    for i, match in enumerate(
        chat["matches"],
        start=1
    ):

        st.write(
            f"{i}. {match['question']} "
            f"({match['score']})"
        )
# ---------- FOOTER ----------

st.markdown("""
<div class="footer">
Built using Python • NLP • TF-IDF • Cosine Similarity • Streamlit
</div>
""", unsafe_allow_html=True)