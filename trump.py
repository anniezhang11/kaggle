import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import nltk
from nltk.stem.porter import *
from wordcloud import WordCloud

# open files
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# clean up data
combi = train.append(test, ignore_index=True)
combi["tidy_tweet"] = np.array(combi["text"])
combi["tidy_tweet"] = combi["tidy_tweet"].str.replace("[^a-zA-Z#]", " ")

# tokenize
tokenized_tweet = combi["tidy_tweet"].apply(lambda x: x.split())

# stem the words
stemmer = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])

# stemming
negative_words = " ".join([text for text in combi["tidy_tweet"][combi["label"] == 1]])
wordcloud = WordCloud(
    width=800, height=500, random_state=21, max_font_size=110
).generate(negative_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

