import re
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import string
import nltk
from nltk.stem.porter import *

# open files
train  = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# clean up data
combi = train.append(test, ignore_index=True)
combi['tidy_tweet'] = np.vectorize(combi['text'])
combi['tidy_tweet'] = combi['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")
combi_new = combi[combi['tidy_tweet'].notnull()]

# tokenize
tokenized_tweet = combi_new['tidy_tweet'].apply(lambda x: x.split())

# stem the words
stemmer = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
