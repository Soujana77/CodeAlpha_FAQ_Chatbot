import streamlit as st

from chatbot import (
    initialize_chatbot,
    get_answer
)

st.set_page_config(
    page_title="AI FAQ Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI FAQ Chatbot")

st.write(
    "Ask questions about AI, Programming, Machine Learning, NLP, Python, Data Science, and more."
)

# Initialize chatbot
faq_df, vectorizer, tfidf_matrix = initialize_chatbot()

# Chat history storage
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
user_question = st.text_input(
    "Enter your question:"
)

# Ask button
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

    else:
        st.warning(
            "Please enter a question."
        )

# Display chat history
st.subheader("💬 Chat History")

for chat in reversed(st.session_state.chat_history):

    st.markdown(
        f"**🧑 You:** {chat['question']}"
    )

    st.markdown(
        f"**🤖 Bot:** {chat['answer']}"
    )

    st.caption(
        f"{chat['label']} | Confidence Score: {chat['confidence']}"
    )

    st.divider()