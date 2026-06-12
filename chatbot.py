import string
import nltk

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

sample = "Can you tell me about Python Programming?"

print(preprocess_text(sample))