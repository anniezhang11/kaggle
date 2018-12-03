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
train["tidy_tweet"] = train["tidy_tweet"].str.replace("[^a-zA-Z#]", " ")

# tokenize
tokenized_train = train["tidy_tweet"].apply(lambda x: x.split())

# stem the words
stemmer = PorterStemmer()
tokenized_train= tokenized_train.apply(lambda x: [stemmer.stem(i) for i in x])

# stemming
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
HT_android = hashtag_extract(train['tidy_tweet'][train['label'] == 1])
HT_iphone = hashtag_extract(train['tidy_tweet'][train['label'] == -1])
HT_android = sum(HT_android,[])
HT_iphone = sum(HT_iphone,[])

# testing data
test = pd.read_csv("test.csv")
test["tidy_tweet"] = np.array(test["text"])
test["tidy_tweet"] = test["tidy_tweet"].str.replace("[^a-zA-Z#]", " ")
tokenized_test = test["tidy_tweet"].apply(lambda x: x.split())
tokenized_test= tokenized_test.apply(lambda x: [stemmer.stem(i) for i in x])
