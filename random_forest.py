import re
import nltk
import pandas as pd
import numpy as np
from nltk.stem.porter import *

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# xTr, yTr = loadData("train.csv", True)
data = pd.read_csv("train.csv")
testdata = pd.read_csv("test.csv")
vectorizer = CountVectorizer()
xTr = vectorizer.fit_transform(data["text"])

xTe = vectorizer.transform(testdata["text"])
print(xTe.shape)
yTr = data["label"]
clf = RandomForestClassifier(n_estimators=15)
# scores = cross_val_score(clf, xTr, yTr, cv=5)
clf = clf.fit(xTr, yTr)
predictions = clf.predict(xTe)
submission = pd.DataFrame()

submission["Label"] = predictions
submission.to_csv("submission.csv")
# print(clf.score(xTr, yTr))

