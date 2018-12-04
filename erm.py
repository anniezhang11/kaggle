import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import nltk
from nltk.stem.porter import *

# from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud

# nltk.download("vader_lexicon")

# functions used
def hashtag_extract(x):
    hashtags = []
    # Loop over the words in the tweet
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)

    return hashtags


# # tokenize the email and hashes the symbols into a vector
# def extractfeaturesnaive(path, B):
#         # initialize all-zeros feature vector
#         v = np.zeros(B)
#         email = femail.read()
#         # breaks for non-ascii characters
#         tokens = email.split()
#         for token in tokens:
#             v[hash(token) % B] = 1
#     return v


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


def adagrad(func, w, alpha, maxiter, eps, delta=1e-02):
    losses = np.zeros(maxiter)
    d, = w.shape
    z = np.zeros(d)
    iteration = 0

    w_old = np.copy(w)
    while iteration < maxiter:
        loss, gradient = func(w_old)

        losses[iteration] = loss
        z += np.square(gradient)

        alphaterm = alpha * gradient / (np.sqrt(z + eps))
        w_new = w_old + alphaterm

        if np.linalg.norm(gradient) < delta:
            w_old = w_new
            break

        w_old = w_new
        iteration += 1

    return w_old, losses


def linclassify(w, xTr):
    return np.sign(xTr.dot(w))


def hinge(w, xTr, yTr, lmbda):
    n, d = xTr.shape

    a = 1 - yTr * xTr.dot(w)
    maximum = np.maximum(a, 0)
    addTerm = lmbda * np.dot(w.T, w)
    loss = np.sum(maximum) + addTerm

    comparison = yTr * np.dot(xTr, w.T)
    gradient = -np.dot((comparison < 1) * yTr, xTr) + 2 * lmbda * w

    return loss, gradient


training_data, training_labels = loadData("train.csv", True)

lmbda = 0.1
_, d = training_data.shape

w, losses = adagrad(
    lambda weight: hinge(weight, training_data, training_labels, lmbda),
    np.random.rand(d),
    1,
    1000,
    1e-06,
)
preds = linclassify(w, training_data)
trainingacc = np.mean(preds == training_labels)
print(trainingacc)

testing_data, testing_labels = loadData("test.csv", False)
preds = linclassify(w, testing_data)


# wordcloud = WordCloud(
#     width=800, height=500, random_state=21, max_font_size=110
# ).generate(android_words)
# plt.figure(figsize=(10, 7))
# plt.imshow(wordcloud, interpolation="bilinear")
# plt.axis("off")
# plt.show()

# processing hashtags
# HT_android = hashtag_extract(train['tidy_tweet'][train['label'] == 1])
# HT_iphone = hashtag_extract(train['tidy_tweet'][train['label'] == -1])
# HT_android = sum(HT_android,[])
# HT_iphone = sum(HT_iphone,[])

# testing data

