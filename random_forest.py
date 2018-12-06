import re
import pandas as pd
import numpy as np

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# nltk.download("vader_lexicon")

from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer

def compute_sentiments(X, data, sentiment_analyzer):
    num_data_points,_ = X.shape

    sentiment_scores = data.apply(lambda x: sentiment_analyzer.polarity_scores(x))
    neg_sentiment_scores = np.zeros(num_data_points)
    neu_sentiment_scores = np.zeros(num_data_points)
    pos_sentiment_scores = np.zeros(num_data_points)
    for idx,score in enumerate(sentiment_scores):
        neg_sentiment_scores[idx] = score["neg"]
        neu_sentiment_scores[idx] = score["neu"]
        pos_sentiment_scores[idx] = score["pos"]
    neg_sentiment_scores = neg_sentiment_scores.reshape(num_data_points, -1)
    neu_sentiment_scores = neu_sentiment_scores.reshape(num_data_points, -1)
    pos_sentiment_scores = pos_sentiment_scores.reshape(num_data_points, -1)

    X = hstack((X,neg_sentiment_scores,neu_sentiment_scores,pos_sentiment_scores))
    return X

split_boundary = 800

# xTr, yTr = loadData("train.csv", True)

data = pd.read_csv("train.csv")
testdata = data[split_boundary:]
data = data[:split_boundary]

vectorizer = CountVectorizer(stop_words='english', min_df=0.02)
xTr = vectorizer.fit_transform(data["text"])
xTe = vectorizer.transform(testdata["text"])
yTr = data["label"]
yTe = testdata["label"]
nTr,_ = xTr.shape
nTe,_ = xTe.shape

# xTe = vectorizer.transform(testdata["text"])
# print(xTe.shape)

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
        np.array(data["retweetCount"]).reshape(nTr, -1),
        np.array(data["favoriteCount"]).reshape(nTr, -1),
    )
)
xTe = hstack(
    (
        xTe,
        np.array(testdata["retweetCount"]).reshape(nTe, -1),
        np.array(testdata["favoriteCount"]).reshape(nTe, -1),
    )
)

clf = RandomForestClassifier(n_estimators=15).fit(xTr, yTr)
print(clf.score(xTe, yTe))
# scores = cross_val_score(clf, xTr, yTr, cv=20)
# print(scores)
# print(scores.mean())

# scores = cross_val_score(clf, xTr, yTr, cv=5)
# print(scores)
# print(scores.mean())

# scores = cross_val_score(clf, xTr, yTr, cv=5)


# print(clf.score(xTr, yTr))

