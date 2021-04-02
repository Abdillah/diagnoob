import pickle

from flask import Flask
from flask import jsonify, request
from flask import render_template

from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

app = Flask("diagnoob")

def preprocess(input):
    # create the tokenizer
    t = pickle.load('./model/001-tokenizer.pickle')

    # Pad the input vectors to ensure a consistent length
    x_input = np.array(t.texts_to_sequences(input))
    return pad_sequences(x_input, maxlen=500)

def postprocess(output):
    f = open('./model/001-labelencoder.pickle', 'rb+')
    enc = pickle.load(f)

    output = output.flatten()
    output_class = np.argmax(output, axis=None, out=None)
    output = [ 1 if k == output_class else 0 for (k, o) in enumerate(output) ]
    decoded = enc.inverse_transform(output)
    return decoded[output_class]


@app.route('/keyword-extract/')
def keyword_extract():
    return render_template('predict.html')


@app.route('/keyword-extract/result', methods = ['POST', 'GET'])
def keyword_extract_result():
    if request.method == 'POST':
        result = request.form
        return render_template('result.html', result = str(result))


@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/api/diagnose/', methods = [ "GET" ])
def diagnose():
    model = keras.models.load_model('./model/001.hdf5')
    input_sentence = request.args.get('sentence')
    preprocessed_input = preprocess(input_sentence)
    output = model.predict(preprocessed_input)
    output = postprocess(output)
    return jsonify(data=output)
