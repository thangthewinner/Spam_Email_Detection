import numpy as np
import streamlit as st
import pickle as pkl
import re
import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

with open('./models/count_vectorizer.pkl', 'rb') as f:
    vectorizer = pkl.load(f)

with open('./models/mnb_model.pkl', 'rb') as f:
    MultinomialNaiveBayes = pkl.load(f)

stop_words = set(stopwords.words('english'))

def preprocess_text(text, stop_words):
    text = text.lower() # Lowercase the text
    text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text) # Remove url
    text = re.sub(r'[^a-z0-9\s]', '', text) # Remove special characters
    text = re.sub(r'\s+', ' ', text) # Remove extra spaces
    text = ' '.join(word for word in text.split() if word not in stop_words) # Remove stopwords
    text = ''.join([i for i in text if not i.isdigit()]) # remove numerics values
    return text

def main():
    model = MultinomialNaiveBayes

    st.title('Spam vs Non-Spam Email Classifier')
    st.subheader('Enter the text')
    
    text = st.text_area('Text')
    if st.button('Predict'):
        text = preprocess_text(text, stop_words)
        X = vectorizer.transform([text]).toarray()
        pred = model.predict(X)
        st.write('Spam' if pred[0] == 1 else 'Non-Spam', 'Email')

if __name__ == '__main__':
    main()

# Congratulations! You have won a lottery ticket worth $1 million. Please click the link to claim your prize.