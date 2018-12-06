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

# xTr, yTr = loadData("train.csv", True)
data = pd.read_csv("train.csv")
vectorizer = CountVectorizer(min_df=0.05)
xTr = vectorizer.fit_transform(data["text"])

# xTe = vectorizer.transform(testdata["text"])
# print(xTe.shape)
yTr = data["label"]
n,_ = xTr.shape

data["tidy_tweet"] = np.array(data["text"])
data["tidy_tweet"] = data["tidy_tweet"].str.lower().replace("[^a-zA-Z#]", " ")

sentiment_analyzer = SentimentIntensityAnalyzer()
sentiment_scores = data["tidy_tweet"].apply(lambda x: sentiment_analyzer.polarity_scores(x))
neg_sentiment_scores = np.zeros(n)
neu_sentiment_scores = np.zeros(n)
pos_sentiment_scores = np.zeros(n)
for idx,score in enumerate(sentiment_scores):
    neg_sentiment_scores[idx] = score["neg"]
    neu_sentiment_scores[idx] = score["neu"]
    pos_sentiment_scores[idx] = score["pos"]
neg_sentiment_scores = neg_sentiment_scores.reshape(n, -1)
neu_sentiment_scores = neu_sentiment_scores.reshape(n, -1)
pos_sentiment_scores = pos_sentiment_scores.reshape(n, -1)
xTr = hstack((xTr,neg_sentiment_scores,pos_sentiment_scores))

clf = RandomForestClassifier(n_estimators=15)
scores = cross_val_score(clf, xTr, yTr, cv=5)
print(scores)
print(scores.mean())

# scores = cross_val_score(clf, xTr, yTr, cv=5)
# clf = clf.fit(xTr, yTr)
# predictions = clf.predict(xTe)
# submission = pd.DataFrame()

# submission["Label"] = predictions
# submission.to_csv("submission.csv")
# print(clf.score(xTr, yTr))

