from flask import Flask
from flask import jsonify
from tensorflow import keras

app = Flask("diagnoob")

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/diagnose/')
def diagnose():
    model = keras.models.load_model('./model/001.hdf5')
    output = model.predict(input);
    return jsonify(data=output)
