import streamlit as st
import string
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()  # Lower case
    text = nltk.word_tokenize(text)  # Tokenization
    text = list(filter(lambda x: x.isalnum(), text))  # Removing special chars
    text = list(filter(lambda x: x not in stopwords.words(
        'english') and x not in string.punctuation, text))  # Removing stop words and punctuation
    text = list(map(ps.stem, text))  # Stemming
    return " ".join(text)


tdidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("SMS Spam Classifier")

input_sms = st.text_area("Enter your SMS")

if st.button('Predict'):
    # 1. Preprocess
    transformed_sms = transform_text(input_sms)
    # 2. Vectorize
    vectorized_sms = tdidf.transform([transformed_sms])
    # 3. Predict
    result = model.predict(vectorized_sms)[0]
    # 4. Display
    if result == 1:
        st.header("SPAM!")
    else:
        st.header("Not spam.")
