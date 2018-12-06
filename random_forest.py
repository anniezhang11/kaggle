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

# xTr, yTr = loadData("train.csv", True)

def convert_to_seconds(time):
    arr = time.split(":")
    return int(arr[0])*360 + int(arr[1])*60
data = pd.read_csv("train.csv")
vectorizer = CountVectorizer(min_df=0.05)
xTr = vectorizer.fit_transform(data["text"])
yTr = data["label"]
nTr, _ = xTr.shape

testdata = pd.read_csv("test.csv")
xTe = vectorizer.transform(testdata["text"])
nTe, _ = xTe.shape

tfidf_transformer = TfidfTransformer()
xTr = tfidf_transformer.fit_transform(xTr)
xTe = tfidf_transformer.fit_transform(xTe)

# data["tidy_tweet"] = np.array(data["text"])
# data["tidy_tweet"] = data["tidy_tweet"].str.lower().replace("[^a-zA-Z#]", " ")

# sentiment_analyzer = SentimentIntensityAnalyzer()
# sentiment_scores = data["tidy_tweet"].apply(
#     lambda x: sentiment_analyzer.polarity_scores(x)
# )
# neg_sentiment_scores = np.zeros(n)
# neu_sentiment_scores = np.zeros(n)
# pos_sentiment_scores = np.zeros(n)
# for idx, score in enumerate(sentiment_scores):
#     neg_sentiment_scores[idx] = score["neg"]
#     neu_sentiment_scores[idx] = score["neu"]
#     pos_sentiment_scores[idx] = score["pos"]
# neg_sentiment_scores = neg_sentiment_scores.reshape(n, -1)
# neu_sentiment_scores = neu_sentiment_scores.reshape(n, -1)
# pos_sentiment_scores = pos_sentiment_scores.reshape(n, -1)

# xTr = hstack((xTr, neg_sentiment_scores, pos_sentiment_scores))
timesTr = np.array([i.split(' ')[1] for i in data["created"]]).reshape(nTr, -1)
secondsTr = np.zeros(nTr)
timesTe = np.array([i.split(' ')[1] for i in testdata["created"]]).reshape(nTe, -1)
secondsTe = np.zeros(nTe)
for i in range(nTr):
    secondsTr[i] = convert_to_seconds(timesTr[i][0])
for i in range(nTe):
    secondsTe[i] = convert_to_seconds(timesTe[i][0])

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

clf = RandomForestClassifier(n_estimators=15)
# scores = cross_val_score(clf, xTr, yTr, cv=100)
# print(scores.mean())
clf = clf.fit(xTr, yTr)
scores = clf.score(xTr, yTr)
print(scores)

predictions = clf.predict(xTe)
submission = pd.DataFrame()
submission["Label"] = predictions
submission.to_csv("submission.csv")

# scores = cross_val_score(clf, xTr, yTr, cv=5)
# print(scores)
# print(scores.mean())

# scores = cross_val_score(clf, xTr, yTr, cv=5)


# print(clf.score(xTr, yTr))

