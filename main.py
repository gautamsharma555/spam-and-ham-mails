import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load Data
@st.cache_data
def load_data():
    data = pd.read_csv("Hamspam.csv", encoding="ISO-8859-1")
    return data

data = load_data()

# Data Cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', ' ', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub('\w*\d\w*', ' ', text)
    text = re.sub('[0-9' ']+', " ", text)
    text = re.sub('[‘’“”…]', ' ', text)
    return text

data['text'] = data['text'].apply(clean_text)

# Remove Stopwords
nltk.download('stopwords')
stop = stopwords.words('english')
data['text'] = data['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

# Vectorization
vectorizer = CountVectorizer(min_df=1, max_df=0.9)
X = vectorizer.fit_transform(data["text"])

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, max_df=0.5, smooth_idf=True)
doc_vec = tfidf_vectorizer.fit_transform(data["text"])

# Split Data
X_train, X_test, y_train, y_test = train_test_split(doc_vec, data['type'], test_size=0.3, random_state=42)

# Train Model
model = MultinomialNB()
model.fit(X_train, y_train)

def predict(text):
    cleaned_text = clean_text(text)
    cleaned_text = " ".join([word for word in cleaned_text.split() if word not in stop])
    transformed_text = tfidf_vectorizer.transform([cleaned_text])
    prediction = model.predict(transformed_text)
    return prediction[0]

# Streamlit UI
st.title("Spam Detection App")
st.write("Enter a message below to check if it's spam or ham.")

input_text = st.text_area("Enter your message:")

if st.button("Check Spam"):
    result = predict(input_text)
    st.write(f"Prediction: {result}")

# Word Cloud
total_text = " ".join(data["text"])
wordcloud = WordCloud(background_color='black', width=800, height=400).generate(total_text)
plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
st.pyplot(plt)