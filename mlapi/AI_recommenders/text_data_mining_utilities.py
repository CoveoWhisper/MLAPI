import pickle
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import re
from collections import Counter

LANGUAGE = 'english'

def parseText(text):
    text = lower_cases(text)
    text = remove_noise(text)
    text = remove_numbers(text)
    text = remove_punctuation(text)
    text = keep_just_letters(text)

    text = remove_stop_words(text)
    text = remove_chatbot_stop_words(text)
    text = do_lemmatization(text)
    text = do_stemming(text)

    return text

def lower_cases(text):
    return text.lower()
def remove_noise(text):
    words = word_tokenize(text)
    noise_sub_words = ['www', '/', '<0x', '@']
    for index in range(len(words)):
       for noise_sub_word in noise_sub_words:
           if noise_sub_word in words[index]:
                words[index] = ''
    return ' '.join([word for word in words if word])
def remove_numbers(text):
    return re.sub(r'\d+', ' ', text)
def remove_punctuation(text):
    return  re.sub(r'\W+|\_', ' ', text)
def keep_just_letters (text):
    return re.sub('[^a-z]+', ' ', text)
def remove_stop_words(text):
    stop_words = stopwords.words(LANGUAGE)
    return ' '.join([i for i in word_tokenize(text) if i not in stop_words])
def remove_chatbot_stop_words(text):
    fichier = open('stopwords/chatbot_stop_words.txt', 'r') # when runing text_data_mining.py
    chatbot_stop_words = fichier.read()
    fichier.close()
    fichier = open('stopwords/stop_words_with_personal_names.txt', 'r') # when runing text_data_mining.py
    stop_words = fichier.read()
    fichier.close()
    stop_words += chatbot_stop_words
    return ' '.join([i for i in word_tokenize(text) if i not in stop_words.split()])
def remove_custom_stop_words(text, custom_stop_words):
    return ' '.join([i for i in word_tokenize(text) if i not in custom_stop_words])
def do_stemming(text):
    stemmer = SnowballStemmer(LANGUAGE)
    return ' '.join([stemmer.stem(i) for i in word_tokenize(text)])
def do_lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(i) for i in word_tokenize(text)])