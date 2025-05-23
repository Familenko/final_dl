import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import spacy


stop_words = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_sm", disable = ['parser','ner'])

def remove_stuff(text):
    text = re.sub("<[^>]*>", " ", text) # Remove html tags
    text = re.sub("\S*@\S*[\s]+", " ", text) # Remove emails
    text = re.sub("https?:\/\/.*?[\s]+", " ", text) # Remove links
    text = re.sub("[^a-zA-Z' ]", "", text) # Remove non-letters
    text = re.sub("[\s]+", " ", text) # Remove excesive whitespaces
    text = text.translate(str.maketrans('', '', string.punctuation)) # Remove punctuation
    
    return text

def remove_stopwords(text, stop_words: set):
    text = text.lower().split()
    text = [word for word in text if not word in stop_words]
    return " ".join(text)

def process_with_stemmer(text):
    stemmer = PorterStemmer()
    text = text.lower().split()
    text = [stemmer.stem(word) for word in text]
    return " ".join(text)

def process_with_lemmatizer(text):
    text = text.lower()
    doc = nlp(text)
    text = " ".join([token.lemma_ for token in doc if len(token.lemma_) > 1 ])

    return text

def replace_words(text, replace_on:dict):
    text = text.lower().split()
    text = [replace_on.get(word) if word in replace_on else word for word in text]
    return " ".join(text)


def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = remove_stuff(text)
    text = remove_stopwords(text, stop_words)
    # text = replace_words(text, contractions)

    # test = process_with_stemmer(text)
    text = process_with_lemmatizer(text)
    
    return text
