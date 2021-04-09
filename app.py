import pickle
import re
import csv
import uuid
import os

from flask import Flask
from flask import jsonify, request
from flask import render_template
from flask_bootstrap import Bootstrap
from werkzeug.wrappers import Response
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt


from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
from io import StringIO
import joblib

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
app = Flask("diagnoob")
Bootstrap(app)


app.config['UPLOADED_PHOTOS_DEST'] = 'static/images'


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


def clean(input):
    input = str(input)
    
    input = decontracted(input)
    input = re.sub("\S*\d\S*", "", input).strip()
    input = re.sub('[^A-Za-z]+', ' ', input)

    return input


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
        result = {
            "trans": request_trans,
            "keywords": list(set(predict(request_trans)[0].split(' ')))
        }
        return render_template('result.html', result=result)


@app.route('/keyword-extract/save', methods = ['POST', 'GET'])
def keyword_extract_save():
    if request.method == 'POST':
        keys = '-'.join(request.form.getlist('keywords'))

        data_trans = clean(request.form['trans'])

        def generate():
            data = StringIO()
            w = csv.writer(data)

            # write header
            w.writerow(('Transcription', 'Keywords'))
            yield data.getvalue()
            data.seek(0)
            data.truncate(0)

            w.writerow((
                data_trans,
                keys
            ))
            yield data.getvalue()
            data.seek(0)
            data.truncate(0)

        # stream the response as the data is generated
        response = Response(generate(), mimetype='text/csv')
        # add a filename
        response.headers.set("Content-Disposition", "attachment", filename="diagno_dataset.csv")
        return response


def predict(input_sentence):
    return model.predict([input_sentence])


@app.route('/')
def hello_world():
    return render_template('index.html')


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


@app.route('/get_word_cloud')
def form_vals():
    trans = request.args.get('trans')
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(width=800, height=800,
                        background_color='white',
                        stopwords=stopwords,
                        min_font_size=10).generate(trans)

    # plot the WordCloud image
    # plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)

    plt.savefig(os.path.join(app.config['UPLOADED_PHOTOS_DEST'], 'wcloud.png')) #save to the images directory

    return jsonify({"result": "<img src='static/images/wcloud.png' width='120' height='90' />"})
