import re
import pandas as pd
import numpy as np

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download("vader_lexicon")

from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

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
vectorizer = CountVectorizer(min_df=0.05)
X = vectorizer.fit_transform(data["text"])
xTr, xTe = X[:split_boundary],X[split_boundary:]
yTr, yTe = data["label"][:split_boundary],data["label"][split_boundary:]
n,_ = xTr.shape

# xTe = vectorizer.transform(testdata["text"])
# print(xTe.shape)

data["tidy_tweet"] = np.array(data["text"])
data["tidy_tweet"] = data["tidy_tweet"].str.lower().replace("[^a-zA-Z#]", " ")

sentiment_analyzer = SentimentIntensityAnalyzer()
xTr = compute_sentiments(xTr, data["tidy_tweet"][:split_boundary], sentiment_analyzer)
xTe = compute_sentiments(xTe, data["tidy_tweet"][split_boundary:], sentiment_analyzer)

clf = RandomForestClassifier(n_estimators=15).fit(xTr, yTr)
print(clf.score(xTe, yTe))
# scores = cross_val_score(clf, xTr, yTr, cv=20)
# print(scores)
# print(scores.mean())

# scores = cross_val_score(clf, xTr, yTr, cv=5)
# clf = clf.fit(xTr, yTr)
# predictions = clf.predict(xTe)
# submission = pd.DataFrame()

# submission["Label"] = predictions
# submission.to_csv("submission.csv")
# print(clf.score(xTr, yTr))

