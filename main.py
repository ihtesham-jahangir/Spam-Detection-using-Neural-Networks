import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from flask import Flask, request, render_template
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense
import pickle

app = Flask(__name__)

# Load the pre-trained model and artifacts
loaded_model = keras.models.load_model("spam_detection_model.h5")
with open("tokenizer.pkl", "rb") as f:
    loaded_tokenizer = pickle.load(f)
with open("max_len.pkl", "rb") as f:
    loaded_max_len = pickle.load(f)

# Function to predict ham or spam using the pre-trained model
def predict_message(message, model, tokenizer, max_len):
    # Preprocess the message
    sequence = tokenizer.texts_to_sequences([message])
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post')
    # Predict probabilities
    probabilities = model.predict(padded_sequence)
    # Convert probabilities to class labels
    prediction = "Ham" if probabilities[0] < 0.5 else "Spam"
    return prediction

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get the message from the form input
        message = request.form['message']
        # Predict using the pre-trained model
        prediction = predict_message(message, loaded_model, loaded_tokenizer, loaded_max_len)
        return render_template('index.html', prediction=prediction)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
