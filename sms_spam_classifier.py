import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

# Initialize the Porter Stemmer
ps = PorterStemmer()

# Ensure you download the necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')


# Function to preprocess text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

        
# Load the pre-trained models
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit App Title
st.title("SMS/Email Spam Classifier")

# Input box for user to enter the SMS/email
input_sms = st.text_input("Enter the message")

if st.button("Classify"):
    # 1. Preprocess the input
    transformed_sms = transform_text(input_sms)

    # 2. Vectorize the input
    vector_input = tfidf.transform([transformed_sms])

    # 3. Predict using the loaded model
    result = model.predict(vector_input)

    # 4. Display the result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
