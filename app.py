import streamlit as st

from chatbot import (
    initialize_chatbot,
    get_answer
)

st.set_page_config(
    page_title="AI FAQ Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize chatbot
faq_df, vectorizer, tfidf_matrix = initialize_chatbot()

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar
with st.sidebar:

    st.title("🤖 AI FAQ Chatbot")

    st.markdown("---")

    st.subheader("📌 About")

    st.write(
        """
        This chatbot answers AI and Programming related questions using:

        - NLP
        - TF-IDF Vectorization
        - Cosine Similarity
        - Streamlit
        """
    )

    st.markdown("---")

    st.metric(
        "Knowledge Base Size",
        len(faq_df)
    )

    if st.button("🗑️ Clear Chat"):

        st.session_state.chat_history = []

        st.rerun()

# Main Page
st.title("🤖 AI FAQ Chatbot")

st.write(
    "Ask questions related to AI, Machine Learning, Python, NLP, Data Science, and Programming."
)

user_question = st.text_input(
    "Ask a question:"
)

if st.button("Ask"):

    if user_question.strip():

        answer, score = get_answer(
            user_question,
            faq_df,
            vectorizer,
            tfidf_matrix
        )

        confidence = round(score, 2)

        if confidence >= 0.80:
            confidence_label = "🟢 High Confidence"
        elif confidence >= 0.50:
            confidence_label = "🟡 Medium Confidence"
        else:
            confidence_label = "🔴 Low Confidence"

        st.session_state.chat_history.append(
            {
                "question": user_question,
                "answer": answer,
                "confidence": confidence,
                "label": confidence_label
            }
        )

# Chat Display
st.subheader("💬 Conversation")

for chat in reversed(st.session_state.chat_history):

    with st.container():

        st.markdown(
            f"### 🧑 You\n{chat['question']}"
        )

        st.markdown(
            f"### 🤖 Bot\n{chat['answer']}"
        )

        st.caption(
            f"{chat['label']} | Confidence Score: {chat['confidence']}"
        )

        st.markdown("---")

# Footer
st.markdown("---")

st.caption(
    "Built using Python, NLP, TF-IDF, Cosine Similarity, and Streamlit."
)