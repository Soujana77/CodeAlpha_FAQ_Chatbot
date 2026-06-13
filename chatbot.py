import string
import nltk
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

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

faq_df = load_faq_data()

print(faq_df[["question", "processed_question"]])

