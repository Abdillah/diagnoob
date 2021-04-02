import pickle

from flask import Flask
from flask import jsonify, request

from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

app = Flask("diagnoob")

def preprocess(input):
    # create the tokenizer
    t = pickle.load('./model/001-tokenizer.pickle')

    # Pad the input vectors to ensure a consistent length
    x_input = np.array(t.texts_to_sequences(input))
    return pad_sequences(x_input, maxlen=500)

def postprocess(output):
    output_classes = np.argmax(output, axis=None, out=None)
    enc = pickle.load('./model/001-labelencoder.pickle')
    return enc.inverse_transform(output_classes)


@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/api/diagnose/')
def diagnose():
    model = keras.models.load_model('./model/001.hdf5')
    input_sentence = request.form['sentence']
    preprocessed_input = preprocess(input_sentence)
    output = model.predict(preprocessed_input)
    output = postprocess(output)
    return jsonify(data=output)
