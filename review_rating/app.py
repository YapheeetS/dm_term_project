from flask import Flask, request, jsonify, render_template, Blueprint, abort
from flask_cors import CORS
from threading import Timer
from time import sleep
from datetime import datetime
from absl import logging
from datetime import date, timedelta, datetime
import sys
import os
import pandas
import string
import tensorflow as tf
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
import string
from tqdm import tqdm
import numpy as np
import random
import pickle

nltk.download('punkt')
nltk.download('stopwords')


app = Flask(__name__)
CORS(app)

@app.route('/', methods= ["GET", "POST"])
def home():
    return render_template("index.html")

@app.route('/api/vote_review')
def vote_review():

    MAX_SEQUENCE_LENGTH = 100
    MAX_NUM_WORDS = 1000
    EMBEDDING_DIM = 100
    VALIDATION_SPLIT = 0.2

    text = ''
    if (all([x in request.args for x in ['text']])):
        text = request.args['text']

    all_texts = []
    text = text.lower()
    # tokenize
    words = word_tokenize(text)
    # topwords
    words = [w for w in words if w not in stopwords.words('english')]
    # remove punctuation
    words = [w for w in words if w not in string.punctuation]
    # Stemming
    words = [PorterStemmer().stem(w) for w in words]
    all_texts.append(words)

    fn = 'tokenizer.pkl'
    with open(fn, 'rb') as f:
        tokenizer = pickle.load(f)

    x_test = tokenizer.texts_to_sequences(all_texts)
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=MAX_SEQUENCE_LENGTH)

    # load model
    model = tf.keras.models.load_model('gru_model.h5')
    y_pred = model.predict(x_test)

    y_pred = y_pred[0]
    y_pred = y_pred.tolist()
    max_num = max(y_pred)
    max_index = y_pred.index(max_num)

    ret = {'voting': str(max_index)}
    ret = jsonify(ret)
    print(ret)
    return ret


@app.route('/api/vote_balance_review')
def vote_balance_review():

    MAX_SEQUENCE_LENGTH = 100
    MAX_NUM_WORDS = 1000
    EMBEDDING_DIM = 100
    VALIDATION_SPLIT = 0.2

    text = ''
    if (all([x in request.args for x in ['text']])):
        text = request.args['text']

    all_texts = []
    text = text.lower()
    # tokenize
    words = word_tokenize(text)
    # topwords
    words = [w for w in words if w not in stopwords.words('english')]
    # remove punctuation
    words = [w for w in words if w not in string.punctuation]
    # Stemming
    words = [PorterStemmer().stem(w) for w in words]
    all_texts.append(words)

    fn = 'balance_tokenizer.pkl'
    with open(fn, 'rb') as f:
        tokenizer = pickle.load(f)

    x_test = tokenizer.texts_to_sequences(all_texts)
    x_test = tf.keras.preprocessing.sequence.pad_sequences(
        x_test, maxlen=MAX_SEQUENCE_LENGTH)

    # load model
    model = tf.keras.models.load_model('balance_gru_model.h5')
    y_pred = model.predict(x_test)

    y_pred = y_pred[0]
    y_pred = y_pred.tolist()
    max_num = max(y_pred)
    max_index = y_pred.index(max_num)

    ret = {'voting': str(max_index)}
    ret = jsonify(ret)
    print(ret)
    return ret


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    port = 2222 if not len(sys.argv) > 1 else int(sys.argv[1])
    app.run(host="0.0.0.0", port=port, threaded=True, debug=False, use_reloader=False)
