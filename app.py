import pickle
import re

from flask import Flask
from flask import jsonify, request
from flask import render_template
from flask_bootstrap import Bootstrap

from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import joblib

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
app = Flask("diagnoob")
Bootstrap(app)


model = joblib.load('./model/002.pkl')

def decontracted(phrase):   # text pre-processing 
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"@", "" , phrase)         # removal of @
    phrase =  re.sub(r"http\S+", "", phrase)   # removal of URLs
    phrase = re.sub(r"#", "", phrase)          # hashtag processing

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def preprocess(input):
    input = str(input)

    print('input', input)
    
    input = decontracted(input)
    input = re.sub("\S*\d\S*", "", input).strip()
    input = re.sub('[^A-Za-z]+', ' ', input)
    # https://gist.github.com/sebleier/554280
    input = ' '.join(list(e.lower() for e in input.split() if e.lower() not in stopwords.words()))
    input = input.strip()

    # create the tokenizer
    f = open('./model/001-tokenizer.pickle', 'rb+')
    t = pickle.load(f)

    # Pad the input vectors to ensure a consistent length
    print('input', input)
    x_input = np.array(t.texts_to_sequences(input))
    print('x_input', x_input)
    return pad_sequences(x_input, maxlen=500)

def postprocess(output):
    f = open('./model/001-labelencoder.pickle', 'rb+')
    enc = pickle.load(f)

    output = output.flatten()
    output_class = np.argmax(output, axis=None, out=None)
    output = [ 1 if k == output_class else 0 for (k, o) in enumerate(output) ]
    decoded = enc.inverse_transform(output)
    return decoded[output_class]


@app.route('/keyword-extract/', methods = ['POST', 'GET'])
def keyword_extract():
    return render_template('predict.html')


@app.route('/keyword-extract/result', methods = ['POST', 'GET'])
def keyword_extract_result():
    if request.method == 'POST':
        request_trans = request.form['transcription']
        return render_template('result.html', result = set(predict(request_trans)[0].split(' ')))

def predict(input_sentence):
    return model.predict([input_sentence])

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/api/diagnose/', methods = [ "GET" ])
def diagnose():
    model = keras.models.load_model('./model/001.hdf5')
    input_sentence = request.args.get('sentence')
    print('input_sentence', input_sentence)
    preprocessed_input = preprocess(input_sentence)
    output = model.predict(preprocessed_input)
    output = postprocess(output)
    return jsonify(data=output)

@app.route('/api/diagnose/v2/', methods = [ "GET" ])
def diagnose_v2():
    model = joblib.load('./model/002.pkl')
    input_sentence = request.args.get('sentence')
    print('input_sentence', input_sentence)
    output = model.predict([input_sentence])
    # output = postprocess(output)
    return jsonify(data=output[0])