import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the trained model and tokenizer
model = load_model('model.h5')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Vocabulary size
vocab_size = len(tokenizer.word_index) + 1

# Maximum length of input sequence (used during training)
max_len = 56  # Replace with your actual max_len value

# Function to predict the next word iteratively
def predict_next_words(input_text, num_words):
    text = input_text

    for i in range(num_words):
    # tokenize
        token_text = tokenizer.texts_to_sequences([text])[0]
    # padding
        padded_token_text = pad_sequences([token_text], maxlen=56, padding='pre')
    # predict
        pos = np.argmax(model.predict(padded_token_text))

        for word,index in tokenizer.word_index.items():
            if index == pos:
                text = text + " " + word
        
    return text

# Streamlit UI
st.title("Next Word Predictor")

# User input for sentence and number of words to predict
user_input = st.text_input("Type your sentence here:")
num_words = st.number_input("Number of words to predict", min_value=None, max_value=None, value=1, step=1)

# Button to trigger prediction
if st.button('Predict'):
    if user_input:
        predicted_words = predict_next_words(user_input, num_words)
        st.write(f"**Predicted Next Words:** {predicted_words}")
    else:
        st.write("Please enter a sentence.")
