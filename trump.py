import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import nltk
from nltk.stem.porter import *
from wordcloud import WordCloud

# functions used
def hashtag_extract(x):
    hashtags = []
    # Loop over the words in the tweet
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)

    return hashtags


# open files
train = pd.read_csv("train.csv")

# clean up data
train["tidy_tweet"] = np.array(train["text"])
train["tidy_tweet"] = train["tidy_tweet"].str.replace("[^a-zA-Z#]", " ")

# tokenize
tokenized_train = train["tidy_tweet"].apply(lambda x: x.split())


# stem the words
stemmer = PorterStemmer()
tokenized_train = tokenized_train.apply(lambda x: [stemmer.stem(i) for i in x])

android_words = " ".join([text for text in train["tidy_tweet"][train["label"] == 1]])
iphone_words = " ".join([text for text in train["tidy_tweet"][train["label"] == -1]])
wordcloud = WordCloud(
    width=800, height=500, random_state=21, max_font_size=110
).generate(android_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# processing hashtags
HT_android = hashtag_extract(train["tidy_tweet"][train["label"] == 1])
HT_iphone = hashtag_extract(train["tidy_tweet"][train["label"] == -1])
HT_android = sum(HT_android, [])
HT_iphone = sum(HT_iphone, [])

# testing data
test = pd.read_csv("test.csv")
test["tidy_tweet"] = np.array(test["text"])
test["tidy_tweet"] = test["tidy_tweet"].str.replace("[^a-zA-Z#]", " ")
tokenized_test = test["tidy_tweet"].apply(lambda x: x.split())
tokenized_test = tokenized_test.apply(lambda x: [stemmer.stem(i) for i in x])


def sqlsplit(xTr, yTr, weights=[]):
    N, D = xTr.shape
    assert D > 0
    assert N > 1
    if weights == []:
        weights = np.ones(N)
    weights = weights / sum(weights)
    bestloss = np.inf
    feature = np.inf
    cut = np.inf

    sort = np.argsort(xTr, axis=0)
    for i in range(D):
        workingarray = sort[:i]
        xTrFeat = xTr[:, i][workingarray]
        yTrFeat = yTr[workingarray]
        weightFeat = weights[workingarray]

        for j in range(N - 1):
            if xTrFeat[j] != xTrFeat[j + 1]:

                cutoff = xTrFeat[j] + xTr[j + 1]

                QL = np.dot(np.square(yTrFeat[0 : j + 1]), weightFeat[0 : j + 1])
                PL = np.dot(yTrFeat[0 : j + 1], weightFeat[0 : j + 1])
                WL = np.sum(weightFeat[0 : j + 1])

                LossLeft = QL - np.square(PL) / WL

                QR = np.dot(np.square(yTrFeat[j + 1 : N]), weightFeat[j + 1 : N])
                PR = np.dot(yTrFeat[j + 1 : N], weightFeat[j + 1 : N])
                WR = np.sum(weightFeat[j + 1 : N])

                LossRight = QR - np.square(PR) / WR

                cutloss = LossLeft + LossRight
                if cutloss < bestloss:
                    feature = i
                    bestloss = cutloss
                    cut = cutoff

    return feature, cut, bestloss
