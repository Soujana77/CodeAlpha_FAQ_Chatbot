import string
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

stop_words = set(stopwords.words("english"))


def preprocess_text(text):
    """
    Clean and preprocess user text
    """

    text = text.lower()

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    tokens = word_tokenize(text)

    filtered_tokens = [
        word
        for word in tokens
        if word not in stop_words
    ]

    cleaned_text = " ".join(filtered_tokens)

    return cleaned_text


def load_faq_data():
    """
    Load FAQ dataset and preprocess questions
    """

    df = pd.read_csv("faq_data.csv")

    df["processed_question"] = df["question"].apply(
        preprocess_text
    )

    return df


def create_tfidf_vectors(df):
    """
    Create TF-IDF vectors from processed questions
    """

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(
        df["processed_question"]
    )

    return vectorizer, tfidf_matrix


def get_answer(user_question, df, vectorizer, tfidf_matrix):
    """
    Find the best matching FAQ answer
    """

    processed_question = preprocess_text(
        user_question
    )

    user_vector = vectorizer.transform(
        [processed_question]
    )

    similarity_scores = cosine_similarity(
        user_vector,
        tfidf_matrix
    )

    best_match_index = similarity_scores.argmax()

    confidence_score = similarity_scores[0][best_match_index]

    if confidence_score < 0.30:
        return (
            "Sorry, I couldn't find a relevant answer.",
            confidence_score
        )

    answer = df.iloc[best_match_index]["answer"]

    return answer, confidence_score


def chatbot():
    """
    Terminal-based chatbot
    """

    faq_df = load_faq_data()

    vectorizer, tfidf_matrix = create_tfidf_vectors(
        faq_df
    )

    print("\nFAQ Chatbot Started!")
    print("Type 'exit' to quit.\n")

    while True:

        user_question = input("You: ")

        if user_question.lower() == "exit":
            print("Bot: Goodbye!")
            break

        answer, score = get_answer(
            user_question,
            faq_df,
            vectorizer,
            tfidf_matrix
        )

        print(f"Bot: {answer}")
        print(f"Confidence: {round(score, 2)}\n")


if __name__ == "__main__":
    chatbot()