import re
import nltk
import pandas as pd
import numpy as np
from nltk.stem.porter import *

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

def hashtag_extract(x):
    hashtags = []
    # Loop over the words in the tweet
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)

    return hashtags

def hashit(data, B):
    # hash that shit
    v = np.zeros(B)
    for token in data:
        v[hash(token) % B] = 1

    return v

def loadData(filename, istraining, B=512):
    """
    INPUT:
    extractfeatures : function to extract features
    B               : dimensionality of feature space
    path            : the path of folder to be processed
    
    OUTPUT:
    X, Y
    """
    # open files
    data = pd.read_csv(filename)

    # clean up data
    data["tidy_tweet"] = np.array(data["text"])
    data["tidy_tweet"] = data["tidy_tweet"].str.lower().replace("[^a-zA-Z#]", " ")

    # tokenize
    tokenized_data = data["tidy_tweet"].apply(lambda x: x.split())

    # stem the words
    stemmer = PorterStemmer()
    tokenized_data = tokenized_data.apply(lambda x: [stemmer.stem(i) for i in x])
    x = tokenized_data.size

    xs = np.zeros((x, B))
    ys = np.zeros(x)

    if istraining:
        for tweet in range(x):
            xs[tweet, :] = hashit(tokenized_data[tweet], B)
            ys[tweet] = data["label"][tweet]
        return xs, ys
    else:
        for tweet in range(x):
            xs[tweet, :] = hashit(tokenized_data[tweet], B)
        return xs, ys

xTr, yTr = loadData("train.csv", True)
clf = RandomForestClassifier(n_estimators=10)
scores = cross_val_score(clf, xTr, yTr, cv=5)
print(scores.mean())