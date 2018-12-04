import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import nltk
from nltk.stem.porter import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer
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

    # word sets
    # android_words = " ".join([text for text in train["tidy_tweet"][train["label"] == 1]])
    # iphone_words = " ".join([text for text in train["tidy_tweet"][train["label"] == -1]])

    # hash that shit

    # for x in tokenized_data:
    #     v = np.zeros(B)
    #     for token in x:
    #         v[hash(token) % B] = 1

    sentiment_analyzer = SentimentIntensityAnalyzer()
    sentiments = data["tidy_tweet"].apply(
        lambda x: sentiment_analyzer.polarity_scores(x)
    )

    xs = np.zeros((len(data), 4))
    ys = np.zeros(len(data))
    for i in range(len(data["tidy_tweet"])):
        if istraining:
            ys[i] = data["label"][i]
        j = 0
        for _, value in sentiments[i].iteritems():
            xs[i][j] = value
            j += 1
        j = 0
    return xs, ys


training_data, training_labels = loadData("train.csv", True)
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
testing_data, testing_labels = loadData("test.csv", False)


class TreeNode(object):
    """Tree class.
    """

    def __init__(self, left, right, parent, cutoff_id, cutoff_val, prediction):
        self.left = left
        self.right = right
        self.parent = parent
        self.cutoff_id = cutoff_id
        self.prediction = prediction


def sqsplit(xTr, yTr, weights=[]):
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
        workingarray = sort[:, i]
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


def cart(xTr, yTr, depth=np.inf, weights=None):
    """Builds a CART tree.

    The maximum tree depth is defined by "maxdepth" (maxdepth=2 means one split).
    Each example can be weighted with "weights".

    Args:
        xTr:        n x d matrix of data
        yTr:        n-dimensional vector
        maxdepth:   maximum tree depth
        weights:    n-dimensional weight vector for data points

    Returns:
        tree:       root of decision tree
    """
    n, d = xTr.shape
    if weights is None:
        w = np.ones(n) / float(n)
    else:
        w = weights

    allSame = np.all(yTr == yTr[0])
    prediction = np.dot(w, yTr) / np.sum(w)

    if depth <= 1 or allSame:
        result = TreeNode(None, None, None, None, None, prediction)
    else:
        feature, cut, loss = sqsplit(xTr, yTr, w)

        if feature == np.inf:
            return TreeNode(None, None, None, None, None, prediction)

        xL = xTr[xTr[:, feature] <= cut]
        yL = yTr[xTr[:, feature] <= cut]
        wL = w[xTr[:, feature] <= cut]

        xR = xTr[xTr[:, feature] > cut]
        yR = yTr[xTr[:, feature] > cut]
        wR = w[xTr[:, feature] > cut]

        left = cart(xL, yL, depth - 1, wL)
        right = cart(xR, yR, depth - 1, wR)

        result = TreeNode(left, right, None, feature, cut, prediction)
        left.parent = result
        right.parent = result
    return result

    # cutoff_id, cutoff_val, _ = sqsplit(xTr, yTr, weights=w)
    # feature_row = xTr[:, cutoff_id]
    # left_indices = np.nonzero(feature_row <= cutoff_val)[0]
    # right_indices = np.nonzero(feature_row > cutoff_val)[0]

    # tree = TreeNode(None, None, None, cutoff_id, cutoff_val, np.average(yTr))
    # tree.depth = 1

    # child_stack = []
    # left_child = TreeNode(None, None, tree, None, None, None)
    # left_child.depth = 2
    # child_stack.append((left_child, left_indices))
    # right_child = TreeNode(None, None, tree, None, None, None)
    # right_child.depth = 2
    # child_stack.append((right_child, right_indices))

    # if depth > 1:
    #     tree.left = left_child
    #     tree.right = right_child

    # while len(child_stack) > 0:
    #     node, indices = child_stack.pop()
    #     xTr_node = xTr[indices]
    #     yTr_node = yTr[indices]
    #     w_node = w[indices]
    #     node.prediction = np.average(yTr_node)

    #     if len(indices) > 1 and node.depth < depth:
    #         cutoff_id, cutoff_val, _ = sqsplit(xTr_node, yTr_node, weights=w_node)
    #         node.cutoff_val = cutoff_val

    #         # Unable to find a cutoff, so there can be no further branch
    #         if node.cutoff_id == np.inf:
    #             continue

    #         feature_row = xTr_node[:, cutoff_id]

    #         left_indices = indices[np.nonzero(feature_row <= cutoff_val)[0]]
    #         right_indices = indices[np.nonzero(feature_row > cutoff_val)[0]]

    #         node_left_child = TreeNode(None, None, node, None, None, None)
    #         node_left_child.depth = node.depth + 1
    #         child_stack.append((node_left_child, left_indices))
    #         node_right_child = TreeNode(None, None, node, None, None, None)
    #         node_right_child.depth = node.depth + 1
    #         child_stack.append((node_right_child, right_indices))

    #         node.left = node_left_child
    #         node.right = node_right_child

    #     return tree


x = training_data[:, 1:]
root = cart(x, training_labels)

