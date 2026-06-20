import string
import pandas as pd
import numpy as np

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

stop_words = set(stopwords.words("english"))


def preprocess_text(text):
    """
    Clean and preprocess text
    """

    text = str(text).lower()

    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    tokens = word_tokenize(text)

    filtered_tokens = [
        word
        for word in tokens
        if word not in stop_words
    ]

    cleaned_text = " ".join(
        filtered_tokens
    )

    return cleaned_text


def load_faq_data():
    """
    Load FAQ dataset
    """

    df = pd.read_csv(
        "faq_data.csv"
    )

    df["processed_question"] = (
        df["question"].apply(
            preprocess_text
        )
    )

    return df


def create_tfidf_vectors(df):
    """
    Create TF-IDF vectors
    """

    vectorizer = TfidfVectorizer()

    tfidf_matrix = (
        vectorizer.fit_transform(
            df["processed_question"]
        )
    )

    return (
        vectorizer,
        tfidf_matrix
    )


def get_answer(
    user_question,
    df,
    vectorizer,
    tfidf_matrix
):
    """
    Get best answer
    """

    processed_question = (
        preprocess_text(
            user_question
        )
    )

    user_vector = (
        vectorizer.transform(
            [processed_question]
        )
    )

    similarity_scores = (
        cosine_similarity(
            user_vector,
            tfidf_matrix
        )
    )

    best_match_index = (
        similarity_scores.argmax()
    )

    confidence_score = (
        similarity_scores[0][
            best_match_index
        ]
    )

    if confidence_score < 0.30:

        top_indices = np.argsort(
            similarity_scores[0]
        )[::-1][:3]

        suggestions = []

        for idx in top_indices:

            suggestions.append(
                df.iloc[idx][
                    "question"
                ]
            )

        response = (
            "Sorry, I couldn't find an exact answer.\n\n"
            "You may be interested in:\n\n"
        )

        for i, question in enumerate(
            suggestions,
            start=1
        ):

            response += (
                f"{i}. {question}\n"
            )

        return (
            response,
            confidence_score
        )

    answer = (
        df.iloc[
            best_match_index
        ]["answer"]
    )

    return (
        answer,
        confidence_score
    )


def get_top_matches(
    user_question,
    df,
    vectorizer,
    tfidf_matrix,
    top_n=3
):
    """
    Return top matching FAQs
    """

    processed_question = (
        preprocess_text(
            user_question
        )
    )

    user_vector = (
        vectorizer.transform(
            [processed_question]
        )
    )

    similarity_scores = (
        cosine_similarity(
            user_vector,
            tfidf_matrix
        )[0]
    )

    top_indices = (
        np.argsort(
            similarity_scores
        )[::-1][:top_n]
    )

    matches = []

    for idx in top_indices:

        matches.append(
            {
                "question":
                df.iloc[idx][
                    "question"
                ],

                "score":
                round(
                    float(
                        similarity_scores[
                            idx
                        ]
                    ),
                    2
                )
            }
        )

    return matches


def initialize_chatbot():
    """
    Initialize chatbot
    """

    faq_df = (
        load_faq_data()
    )

    (
        vectorizer,
        tfidf_matrix
    ) = create_tfidf_vectors(
        faq_df
    )

    return (
        faq_df,
        vectorizer,
        tfidf_matrix
    )


if __name__ == "__main__":

    faq_df, vectorizer, tfidf_matrix = (
        initialize_chatbot()
    )

    answer, score = get_answer(
        "What is Python?",
        faq_df,
        vectorizer,
        tfidf_matrix
    )

    print(
        "\nAnswer:"
    )

    print(answer)

    print(
        "\nConfidence:",
        round(score, 2)
    )