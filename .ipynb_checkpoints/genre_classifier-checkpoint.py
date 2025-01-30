import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

# Initialize the Porter Stemmer
ps = PorterStemmer()

# Ensure NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('punkt')

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    filtered_words = [
        ps.stem(word) for word in text
        if word not in stopwords.words('english') and word not in string.punctuation
    ]
    return " ".join(filtered_words)

# Load the pre-trained models and vectorizer
tfidf = pickle.load(open('tfidf_vectorizer_genre.pkl', 'rb'))
svm_model = pickle.load(open('svm_model_genre.pkl', 'rb'))
id_to_genre = pickle.load(open('id_to_genre_mapping.pkl', 'rb'))

# Streamlit App Title
st.title("Movie Genre Classifier")

# Input box for user to enter the movie description
input_description = st.text_area("Enter the movie description:")

if st.button("Classify"):
    if input_description.strip() == "":
        st.error("Please enter a valid movie description.")
    else:
        # 1. Preprocess the input
        processed_text = preprocess_text(input_description)

        # 2. Vectorize the input
        vector_input = tfidf.transform([processed_text])

        # 3. Predict using the loaded model
        predicted_genre_id = svm_model.predict(vector_input)[0]

        # 4. Map the prediction to the genre name
        predicted_genre = id_to_genre.get(predicted_genre_id, "Unknown Genre")

        # 5. Display the result
        st.subheader(f"Predicted Genre: {predicted_genre}")
