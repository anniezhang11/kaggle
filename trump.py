import re
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import string
import nltk

train  = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.head()