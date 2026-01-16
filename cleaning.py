import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from string import punctuation

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

words_to_keep = {
    'ok', 'okay',
    'not', 'no', 'nor', 'but',
    'don', "don't", 'ain', 'aren', "aren't", 'couldn', "couldn't",
    'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
    'haven', "haven't", 'isn', "isn't", 'mightn', "mightn't", 'mustn', "mustn't",
    'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't",
    'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
}

contractions_dict = {
    "i'm": "i am",
    "can't": "cannot",
    "won't": "will not",
    "it's": "it is",
    "don't": "do not"
}

happy_pattern = r'(?<!\S)(?:[:;=8][-]?[\)D\]}pP3]|<3)(?!\S)' # POSITIFS : :) :-) :D :] =] ;) :p :P :3 <3
sad_pattern = r'(?<!\S)[:;=8][-]?[\(\[/{|c](?!\S)' # NEGATIFS : :( :-( :[ =[ :/ :{ :| :c

stop_words = set(stopwords.words('english'))
stop_words = stop_words - words_to_keep
punctuation = punctuation + "’"
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def clean_text(text, processing = "lemmatizer"):
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)

    text = re.sub(happy_pattern, 'tokensmileyhappy', text)
    text = re.sub(sad_pattern, 'tokensmileysad', text)

    text = text.lower()

    for key, value in contractions_dict.items():
        text = text.replace(key, value)

    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    text = text.translate(str.maketrans('', '', punctuation))
    text = text.replace("’", '')

    if processing == "stemmer":
        processor = stemmer.stem
    else:
        processor = lemmatizer.lemmatize

    tokens = [
        processor(word)
        for word in word_tokenize(text)
        if word not in stop_words
        and not word.isdigit() # Retirer les mots composés uniquement de chiffres
        and (len(word) > 2 or word in words_to_keep) # Retirer les mots de 2 lettres ou moins
    ]

    return tokens