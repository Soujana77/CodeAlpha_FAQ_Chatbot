import string
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity

stop_words = set(stopwords.words("english"))


def preprocess_text(text):
    """
    Clean and preprocess user text
    """

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    filtered_tokens = [
        word
        for word in tokens
        if word not in stop_words
    ]

    # Join back into string
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

    processed_question = preprocess_text(user_question)

    user_vector = vectorizer.transform(
        [processed_question]
    )

    similarity_scores = cosine_similarity(
        user_vector,
        tfidf_matrix
    )

    best_match_index = similarity_scores.argmax()

    confidence_score = similarity_scores[0][best_match_index]

    answer = df.iloc[best_match_index]["answer"]

    return answer, confidence_score

faq_df = load_faq_data()

vectorizer, tfidf_matrix = create_tfidf_vectors(
    faq_df
)

question = "Tell me about Python"

answer, score = get_answer(
    question,
    faq_df,
    vectorizer,
    tfidf_matrix
)

print("Question:", question)
print("Answer:", answer)
print("Confidence:", round(score, 2))