########################################################################################################################
import firebase_admin
from firebase_admin import credentials, firestore_async, firestore
# from firebase_admin import firestore
import asyncio

# from google.cloud import firestore
import json, jsonify

# Use a service account.
cred = credentials.Certificate('goatapp-b4e08-firebase-adminsdk-gf64v-b9b8b301fd.json')
firebase_admin.initialize_app(cred)

db = firestore_async.client()


i = 0
output = []

async def print_documents():
    global i
    global output   # Initialize an empty list to store the output
    collections = db.collection("Sessions").document("867946").collections()
    async for collection in collections:
        async for doc in collection.stream():
            output.append(doc.get('body'))  # Append each line to the output list
            print(output[i])
            i += 1




# Run the event loop using asyncio.run()
asyncio.run(print_documents())
print("output: ", output)
########################################################################################################################
import json
from flask import Flask
from flask_cors import CORS, cross_origin
import pandas as pd
import nltk

nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, \
    f1_score
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
import numpy as np
import string

df = pd.read_csv('Twitter_Data.csv')

print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")

print(df.head())

# Preprocess data
stopwords_list = stopwords.words('english') + list(string.punctuation)


# Function to preprocess text
def preprocess_text(text):
    if pd.isnull(text):
        return ""
    else:
        text = text.lower()
        text = ''.join(c for c in text if c.isalnum() or c.isspace())
        tokens = word_tokenize(text)
        filtered_tokens = [token for token in tokens if token not in stopwords_list]
        preprocessed_text = ' '.join(filtered_tokens)
        return preprocessed_text


df['clean_text'] = df['clean_text'].apply(preprocess_text)

df.dropna(subset=['clean_text', 'category'], inplace=True)

print(df[['clean_text', 'category']].head())

# Feature extraction: Convert text into numerical representation using TF-IDF vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text'])
y = df['category']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print('---------------------------------')
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)
print('---------------------------------')


# new_texts = [
#     'LFC have won 30 of their 34 Premier League games this season - the fastest any side has ever reached 30 wins in a sea',
#     'I love you',
#     'this is awesome',
#     'I hate this at all'
# ]
# new_texts = [print_documents()]
# print(new_texts)
# async def main():
#     new_texts = [
#         'LFC have won 30 of their 34 Premier League games this season - the fastest any side has ever reached 30 wins in a sea',
#         'I love you',
#         'this is awesome',
#         'I hate this at all'
#     ]
#
#     output = await print_documents()
#     new_texts.extend(output)  # Add elements from output to new_texts
#
#     print(new_texts)
#
# new_texts = [main()]
# new_texts = asyncio.run(print_documents())
# async def main():
#     new_texts = await print_documents()
#     print(new_texts)
#
# asyncio.run(main())

# loop = asyncio.get_event_loop()
# new_texts = [
#     'LFC have won 30 of their 34 Premier League games this season - the fastest any side has ever reached 30 wins in a sea',
#     'I love you',
#     'this is awesome',
#     'I hate this at all'
# ]
#
# output = loop.run_until_complete(print_documents())
# new_texts.append(output)
# print(new_texts)

# async def main():
#     new_texts = [
#         'LFC have won 30 of their 34 Premier League games this season - the fastest any side has ever reached 30 wins in a sea',
#         'I love you',
#         'this is awesome',
#         'I hate this at all'
#     ]
#
#     output = await print_documents()
#     new_texts.extend(output)  # Add elements from output to new_texts
#
#     print(new_texts)
#
# # Create and run the event loop
# loop = asyncio.get_event_loop()
# try:
#     loop.run_until_complete(main())
# finally:
#     loop.close()
# # Run the event loop using asyncio.run()
# new_texts = asyncio.run(main())

# async def main():
#     new_text = [
#         'LFC have won 30 of their 34 Premier League games this season - the fastest any side has ever reached 30 wins in a sea',
#         'I love you',
#         'this is awesome',
#         'I hate this at all'
#     ]
#     output = await print_documents()
#     new_text.extend(output)  # Add elements from output to new_texts
#     print(new_text)
#
#
# def run_async_code():
#     asyncio.run(main())
#
#
# # Invoke the function to run the asynchronous code
# new_texts = run_async_code()

# new_texts = [
#     'LFC have won 30 of their 34 Premier League games this season - the fastest any side has ever reached 30 wins in a sea',
#     'I love you',
#     'this is awesome',
#     'I hate this at all'
# ]
new_texts = output
print(new_texts)

preprocessed_new_texts = [preprocess_text(text) for text in new_texts]
X_new = vectorizer.transform(preprocessed_new_texts)
y_new_pred = clf.predict(X_new)

sentiment_mapping = {1.0: 'positive', -1.0: 'negative', 0.0: 'natural'}


def get_predicted_sentiments():
    predicted_sentiments = []
    for i in range(len(new_texts)):
        predicted_sentiment = sentiment_mapping[y_new_pred[i]]
        predicted_sentiments.append(predicted_sentiment)

    sentiment_counts = {}
    total_sentiments = len(predicted_sentiments)

    for sentiment in predicted_sentiments:
        if sentiment in sentiment_counts:
            sentiment_counts[sentiment] += 1
        else:
            sentiment_counts[sentiment] = 1

    print("Sentiment percentages:")
    for sentiment, count in sentiment_counts.items():
        percentage = (count / total_sentiments) * 100
        print(f"{sentiment}: {percentage:.2f}%")
        return percentage

    return predicted_sentiments



async def add_document():
    # Access Firestore
    db = firestore.client()
    
    # Define the data to be added
    data = {
        'sentiment': '25.0'
    }
    
    try:
        # Add the document to the "Sessions" collection
        await db.collection('Sessions').document("867946").set(data, merge=True)
        print('Document added successfully.')
    except Exception as e:
        print(f'Error adding document: {e}')

# Execute the async function
asyncio.run(add_document())


########################################################################################################################

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route("/")
@cross_origin()
def helloWorld():
    sentiments = get_predicted_sentiments()
    return json.dumps(sentiments)


app.run()
########################################################################################################################
