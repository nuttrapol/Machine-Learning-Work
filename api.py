import flask
from flask import request, jsonify
from pythainlp.tokenize import word_tokenize
import pickle
# ใช้ตัดคำภาษาไทย
import deepcut

# ใช้งาน regex
import re

# จัดการเกี่ยวกับ array
import numpy as np

import random
from sklearn.model_selection import train_test_split
import numpy
import nltk
from nltk import FreqDist, precision, recall, f_measure, NaiveBayesClassifier
from nltk.classify import apply_features
from nltk.classify import util
from sklearn.metrics import accuracy_score
import collections, itertools
import logging
import pandas as pd
from numpy import random
import gensim
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from pythainlp.tokenize import word_tokenize
from pythainlp.tag import pos_tag
from nltk.tokenize import RegexpTokenizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import statistics 
from statistics import mode 
import time

app = flask.Flask(__name__)
app.config["DEBUG"] = True


books = [
    {'id': 0,
     'title': 'A Fire Upon the Deep',
     'author': 'Vernor Vinge',
     'first_sentence': 'The coldsleep itself was dreamless.',
     'year_published': '1992'},
    {'id': 1,
     'title': 'The Ones Who Walk Away From Omelas',
     'author': 'Ursula K. Le Guin',
     'first_sentence': 'With a clamor of bells that set the swallows soaring, the Festival of Summer came to the city Omelas, bright-towered by the sea.',
     'published': '1973'},
    {'id': 2,
     'title': 'Dhalgren',
     'author': 'Samuel R. Delany',
     'first_sentence': 'to wound the autumnal city.',
     'published': '1975'}
]


@app.route('/', methods=['GET'])
def home():
    return '''<h1>Distant Reading Archive</h1>
<p>A prototype API for distant reading of science fiction novels.</p>'''


@app.route('/api/analyze', methods=['GET'])
def analyze():
    if 'text' in request.args:
        text = request.args.get('text')
        print(text)
    else:
        return '{ success: "false", error : "No text field provided. Please specify an text" }'  

    # preprocees the input sentence
    start_time = time.time()
    cut_sentence=word_tokenize(text.lower() ,engine="newmm")

    with open('C:/Users/nuttapol/Documents/wisesightHW/sentiment_tokenizer.pickle', 'rb') as handle:
    	tokenizer = pickle.load(handle)
    	print('@@@@@@')

    sequences = tokenizer.texts_to_sequences(cut_sentence)
    text_seq = pad_sequences(sequences, maxlen=500)

    #load deep learning model
    model = load_model('C:/Users/nuttapol/Documents/wisesightHW/best_sentiment_model.h5')
    y_pred = model.predict(text_seq)
    Y_pred = np.argmax(y_pred, axis=1)

    label = mode(Y_pred)
    
    if label == 0:
    	result = 'positive'
    elif label == 1:
    	result = 'negative'
    elif label == 2:
    	result = 'neutral'



    final_result = '{ success: "true", sentiment : "'+result+'" }'

    # Use the jsonify function from Flask to convert our list of
    # Python dictionaries to the JSON format.
    end_time = time.time()
    print(end_time - start_time)
    return final_result


@app.route('/api/v1/resources/books', methods=['GET'])
def api_id():
    # Check if an ID was provided as part of the URL.
    # If ID is provided, assign it to a variable.
    # If no ID is provided, display an error in the browser.
    if 'id' in request.args:
        id = int(request.args['id'])
    else:
        return "Error: No id field provided. Please specify an id."

    # Create an empty list for our results
    results = []

    # Loop through the data and match results that fit the requested ID.
    # IDs are unique, but other fields might return many results
    for book in books:
        if book['id'] == id:
            results.append(book)

    # Use the jsonify function from Flask to convert our list of
    # Python dictionaries to the JSON format.
    return jsonify(results)

app.run()

