
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocessor(text):
    # Removing special characters and digits
    letters_only = re.sub("[^a-zA-Z]", " ", text)
    # change sentence to lower case
    letters_only = letters_only.lower()
    # tokenize into words
    words = letters_only.split()
    # remove stop words
    words = [word for word in words if word not in stopwords.words("english")]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)
