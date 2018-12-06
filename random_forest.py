import re
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer

# function outputs sentiment data based off of tweet text
def compute_sentiments(X, data, sentiment_analyzer):
    num_data_points, _ = X.shape

    sentiment_scores = data.apply(lambda x: sentiment_analyzer.polarity_scores(x))
    neg_sentiment_scores = np.zeros(num_data_points)
    neu_sentiment_scores = np.zeros(num_data_points)
    pos_sentiment_scores = np.zeros(num_data_points)
    for idx, score in enumerate(sentiment_scores):
        neg_sentiment_scores[idx] = score["neg"]
        neu_sentiment_scores[idx] = score["neu"]
        pos_sentiment_scores[idx] = score["pos"]
    neg_sentiment_scores = neg_sentiment_scores.reshape(num_data_points, -1)
    neu_sentiment_scores = neu_sentiment_scores.reshape(num_data_points, -1)
    pos_sentiment_scores = pos_sentiment_scores.reshape(num_data_points, -1)

    X = hstack((X, neg_sentiment_scores, neu_sentiment_scores, pos_sentiment_scores))
    return X

# function converts time created string into number of seconds past midnight
def convert_to_seconds(time):
    arr = time.split(":")
    return int(arr[0]) * 360 + int(arr[1]) * 60

# initializations
data = pd.read_csv("train.csv")
testdata = pd.read_csv("test.csv")
split_boundary = 800

# vectorize text from tweets
vectorizer = CountVectorizer(min_df=0.05, ngram_range=(1, 20), analyzer='char_wb')
xTr = vectorizer.fit_transform(data["text"])
xTe = vectorizer.transform(testdata["text"])

yTr = data["label"]
nTr, _ = xTr.shape
nTe, _ = xTe.shape

# apply tfidf to tweet words
tfidf_transformer = TfidfTransformer()
xTr = tfidf_transformer.fit_transform(xTr)
xTe = tfidf_transformer.fit_transform(xTe)

# process time created data
timesTr = np.array([i.split(" ")[1] for i in data["created"]]).reshape(nTr, -1)
secondsTr = np.zeros(nTr)
timesTe = np.array([i.split(" ")[1] for i in testdata["created"]]).reshape(nTe, -1)
secondsTe = np.zeros(nTe)
for i in range(nTr):
    secondsTr[i] = convert_to_seconds(timesTr[i][0])
for i in range(nTe):
    secondsTe[i] = convert_to_seconds(timesTe[i][0])

# Add sentiments
sentiment_analyzer = SentimentIntensityAnalyzer()
data["tidy_tweet"] = np.array(data["text"])
data["tidy_tweet"] = data["tidy_tweet"].str.lower().replace("[^a-zA-Z#]", " ")
xTr = compute_sentiments(xTr, data["tidy_tweet"], sentiment_analyzer)

testdata["tidy_tweet"] = np.array(testdata["text"])
testdata["tidy_tweet"] = testdata["tidy_tweet"].str.lower().replace("[^a-zA-Z#]", " ")
xTe = compute_sentiments(xTe, testdata["tidy_tweet"], sentiment_analyzer)

# Add retweet and feature counts
xTr = hstack(
    (
        xTr,
        np.array(secondsTr).reshape(nTr, -1),
        np.array(data["retweetCount"]).reshape(nTr, -1),
        np.array(data["favoriteCount"]).reshape(nTr, -1),
    )
)
xTe = hstack(
    (
        xTe,
        np.array(secondsTe).reshape(nTe, -1),
        np.array(testdata["retweetCount"]).reshape(nTe, -1),
        np.array(testdata["favoriteCount"]).reshape(nTe, -1),
    )
)

# build model and generate predictions
clf = RandomForestClassifier(n_estimators=20).fit(xTr, yTr)

predictions = clf.predict(xTe)
submission = pd.DataFrame()
submission["Label"] = predictions
submission.to_csv("submission.csv")
