import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import nltk
from nltk.stem.porter import *
from wordcloud import WordCloud

#functions used
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
train["tidy_tweet"] = train["tidy_tweet"].str.lower().replace("[^a-zA-Z#]", " ")

# tokenize
tokenized_train = train["tidy_tweet"].apply(lambda x: x.split())

# stem the words
stemmer = PorterStemmer()
tokenized_train= tokenized_train.apply(lambda x: [stemmer.stem(i) for i in x])

# stemming
android_words = " ".join([text for text in train["tidy_tweet"][train["label"] == 1]])
iphone_words = " ".join([text for text in train["tidy_tweet"][train["label"] == -1]])
# wordcloud = WordCloud(
#     width=800, height=500, random_state=21, max_font_size=110
# ).generate(android_words)
# plt.figure(figsize=(10, 7))
# plt.imshow(wordcloud, interpolation="bilinear")
# plt.axis("off")
# plt.show()

# processing hashtags
HT_android = hashtag_extract(train['tidy_tweet'][train['label'] == 1])
HT_iphone = hashtag_extract(train['tidy_tweet'][train['label'] == -1])
HT_android = sum(HT_android,[])
HT_iphone = sum(HT_iphone,[])
print(HT_android)

# testing data
test = pd.read_csv("test.csv")
test["tidy_tweet"] = np.array(test["text"])
test["tidy_tweet"] = test["tidy_tweet"].str.replace("[^a-zA-Z#]", " ")
tokenized_test = test["tidy_tweet"].apply(lambda x: x.split())
tokenized_test= tokenized_test.apply(lambda x: [stemmer.stem(i) for i in x])

class TreeNode(object):
    """Tree class.
    """

    def __init__(self, left, right, parent, cutoff_id, cutoff_val, prediction):
        self.left = left
        self.right = right
        self.parent = parent
        self.cutoff_id = cutoff_id
        self.prediction = prediction


def cart(xTr, yTr, depth=np.inf, weights = None):
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
    n,d = xTr.shape
    if weights is None:
        w = np.ones(n) / float(n)
    else:
        w = weights
    
    cutoff_id, cutoff_val, _ = sqsplit(xTr, yTr, weights=w)
    feature_row = xTr[:,cutoff_id]
    left_indices = np.nonzero(feature_row <= cutoff_val)[0]
    right_indices = np.nonzero(feature_row > cutoff_val)[0]

    tree = TreeNode(None, None, None, cutoff_id, cutoff_val, np.average(yTr))
    tree.depth = 1

    child_stack = []
    left_child = TreeNode(None, None, tree, None, None, None)
    left_child.depth = 2
    child_stack.append((left_child, left_indices))
    right_child = TreeNode(None, None, tree, None, None, None)
    right_child.depth = 2
    child_stack.append((right_child, right_indices))

    if depth > 1:
        tree.left = left_child
        tree.right = right_child

    while len(child_stack) > 0:
        node, indices = child_stack.pop()
        xTr_node = xTr[indices]
        yTr_node = yTr[indices]
        w_node = w[indices]
        node.prediction = np.average(yTr_node)

        if len(indices) > 1 and node.depth < depth:
            cutoff_id, cutoff_val, _ = sqsplit(xTr_node, yTr_node, weights = w_node)
            node.cutoff_val = cutoff_val

            # Unable to find a cutoff, so there can be no further branch
            if node.cutoff_id == np.inf:
                continue

            feature_row = xTr_node[:,cutoff_id]

            left_indices = indices[np.nonzero(feature_row <= cutoff_val)[0]]
            right_indices = indices[np.nonzero(feature_row > cutoff_val)[0]]

            node_left_child = TreeNode(None, None, node, None, None, None)
            node_left_child.depth = node.depth + 1
            child_stack.append((node_left_child, left_indices))
            node_right_child = TreeNode(None, None, node, None, None, None)
            node_right_child.depth = node.depth + 1
            child_stack.append((node_right_child, right_indices))

            node.left = node_left_child
            node.right = node_right_child

        return tree
