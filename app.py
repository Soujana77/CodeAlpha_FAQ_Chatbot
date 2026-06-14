import streamlit as st

from chatbot import (
    initialize_chatbot,
    get_answer
)

st.set_page_config(
    page_title="FAQ Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI FAQ Chatbot")

st.write(
    "Ask a question from the FAQ knowledge base."
)

faq_df, vectorizer, tfidf_matrix = initialize_chatbot()

user_question = st.text_input(
    "Enter your question:"
)

if st.button("Ask"):

    if user_question.strip():

        answer, score = get_answer(
            user_question,
            faq_df,
            vectorizer,
            tfidf_matrix
        )

        st.success(answer)

        st.write(
            f"Confidence Score: {round(score, 2)}"
        )

    else:

        st.warning(
            "Please enter a question."
        )