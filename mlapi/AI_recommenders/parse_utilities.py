import pickle
from nltk import download
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import re
from collections import Counter

LANGUAGE = 'english'
noise_sub_words = ['www', '/', '<0x', '@']

download('wordnet')
download('punkt')
download('stopwords')

with open('stopwords/chatbot_stop_words.txt', 'r') as chatbot_stop_words, open('stopwords/stop_words_with_personal_names.txt', 'r') as names_stop_words:
    stop_words = set(chatbot_stop_words.read())
    stop_words.update(names_stop_words.read())
    stop_words.update(stopwords.words(LANGUAGE))


def parseText(text):
    text = text.lower()
    text = remove_noise(text)
    text = keep_just_letters(text)
    #print(text)
    text = remove_stop_words(text)
    #print("remove stop words : %s" % text)
    text = do_lemmatization(text)
    #print("Lemmatization : %s" % text)
    text = do_stemming(text)
    #print("Stemming: %s" % text)

    return text


def remove_noise(text):
    words = word_tokenize(text)
    return ' '.join([word for word in words if not any(noise_sub_word in word for noise_sub_word in noise_sub_words)])


def remove_numbers(text):
    return re.sub(r'\d+', ' ', text)


def remove_punctuation(text):
    return re.sub(r'\W+|\_', ' ', text)


def keep_just_letters(text):
    return re.sub('[^a-z]+', ' ', text)


def remove_stop_words(text):
    stop_words = stopwords.words(LANGUAGE)
    return ' '.join([i for i in word_tokenize(text) if i not in stop_words])


def remove_custom_stop_words(text, custom_stop_words):
    return ' '.join([i for i in word_tokenize(text) if i not in custom_stop_words])


def do_stemming(text):
    stemmer = SnowballStemmer(LANGUAGE)
    return ' '.join([stemmer.stem(i) for i in word_tokenize(text)])


def do_lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(i) for i in word_tokenize(text)])


def get_word_mapping(text):
    query_words = dict()
    for word in text.split():
        parsed_word = parseText(word)
        if parsed_word or word.isdigit() or word[1:].replace(".", "", 1).isdigit():
            query_words[word] = parsed_word

    unique_query_words = dict()
    parsed_word_set = set()
    for word, parsed_word in query_words.items():
        split_parsed_word = parsed_word.split()
        if len(split_parsed_word) > 1:
            parsed_word = split_parsed_word[0]

        if parsed_word not in parsed_word_set or parsed_word == '':
            parsed_word_set.add(parsed_word)
            unique_query_words[word] = parsed_word

    return unique_query_words
