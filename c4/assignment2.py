
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.0** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-text-mining/resources/d9pwm) course resource._
# 
# ---

# # Assignment 2 - Introduction to NLTK
# 
# In part 1 of this assignment you will use nltk to explore the Herman Melville novel Moby Dick. Then in part 2 you will create a spelling recommender function that uses nltk to find words similar to the misspelling. 

# ## Part 1 - Analyzing Moby Dick

# In[2]:

import nltk
import pandas as pd
import numpy as np

# If you would like to work with the raw text you can use 'moby_raw'
with open('moby.txt', 'r') as f:
    moby_raw = f.read()
    
# If you would like to work with the novel in nltk.Text format you can use 'text1'
moby_tokens = nltk.word_tokenize(moby_raw)
text1 = nltk.Text(moby_tokens)


# ### Example 1
# 
# How many tokens (words and punctuation symbols) are in text1?
# 
# *This function should return an integer.*

# In[2]:

def example_one():
    
    return len(nltk.word_tokenize(moby_raw)) # or alternatively len(text1)

# example_one()


# ### Example 2
# 
# How many unique tokens (unique words and punctuation) does text1 have?
# 
# *This function should return an integer.*

# In[3]:

def example_two():
    
    return len(set(moby_tokens)) # or alternatively len(set(text1))

 #example_two()


# ### Example 3
# 
# After lemmatizing the verbs, how many unique tokens does text1 have?
# 
# *This function should return an integer.*

# In[4]:

from nltk.stem import WordNetLemmatizer

def example_three():

    return 0 # len(set(lemmatized))

# example_three()


# ### Question 1
# 
# What is the lexical diversity of the given text input? (i.e. ratio of unique tokens to the total number of tokens)
# 
# *This function should return a float.*

# In[15]:

def answer_one():
    t = moby_tokens
    return 1. * len(set(t)) / len(t)
    
answer_one()


# ### Question 2
# 
# What percentage of tokens is 'whale'or 'Whale'?
# 
# *This function should return a float.*

# In[19]:

def answer_two():
    return 100. * len([w for w in moby_tokens if w == 'whale' or w == 'Whale']) / float(len(moby_tokens))

answer_two()


# ### Question 3
# 
# What are the 20 most frequently occurring (unique) tokens in the text? What is their frequency?
# 
# *This function should return a list of 20 tuples where each tuple is of the form `(token, frequency)`. The list should be sorted in descending order of frequency.*

# In[28]:

def answer_three():
    dist = nltk.FreqDist(moby_tokens)
    return dist.most_common()[:20]

answer_three()


# ### Question 4
# 
# What tokens have a length of greater than 5 and frequency of more than 150?
# 
# *This function should return a sorted list of the tokens that match the above constraints. To sort your list, use `sorted()`*

# In[32]:

def answer_four():
    t = moby_tokens
    dist = nltk.FreqDist(t)
    ws = set([w for w in t if dist[w]>150 and len(w)>5])
    
    return sorted(ws)

answer_four()


# ### Question 5
# 
# Find the longest word in text1 and that word's length.
# 
# *This function should return a tuple `(longest_word, length)`.*

# In[35]:

def answer_five():
    t = set(moby_tokens)
    ls = [(w, len(w)) for w in t]
    return sorted(ls, key=lambda t: t[1], reverse=True)[0]
    
answer_five()


# ### Question 6
# 
# What unique words have a frequency of more than 2000? What is their frequency?
# 
# "Hint:  you may want to use `isalpha()` to check if the token is a word and not punctuation."
# 
# *This function should return a list of tuples of the form `(frequency, word)` sorted in descending order of frequency.*

# In[36]:

def answer_six():
    t = moby_tokens
    ws = [w for w in t if w.isalpha()]
    dist = nltk.FreqDist(ws)
    res = [(dist[w], w) for w in dist.keys() if dist[w]>2000]
    return sorted(res, key=lambda w: w[0], reverse=True)

answer_six()

# ### Question 7
# 
# What is the average number of tokens per sentence?
# 
# *This function should return a float.*

# In[39]:

def answer_seven():
    
    return np.mean([len(nltk.word_tokenize(s)) for s in nltk.sent_tokenize(moby_raw)])
    
answer_seven()


# ### Question 8
# 
# What are the 5 most frequent parts of speech in this text? What is their frequency?
# 
# *This function should return a list of tuples of the form `(part_of_speech, frequency)` sorted in descending order of frequency.*

# In[ ]:

def answer_eight():
    ps = [pos[1] for pos in nltk.pos_tag(moby_tokens)]
    fps = nltk.FreqDist(ps).most_common()[:5]
    
    return sorted(fps, key=lambda t: t[1], reverse=True)

answer_eight()

# ## Part 2 - Spelling Recommender
# 
# For this part of the assignment you will create three different spelling recommenders, that each take a list of misspelled words and recommends a correctly spelled word for every word in the list.
# 
# For every misspelled word, the recommender should find find the word in `correct_spellings` that has the shortest distance*, and starts with the same letter as the misspelled word, and return that word as a recommendation.
# 
# *Each of the three different recommenders will use a different distance measure (outlined below).
# 
# Each of the recommenders should provide recommendations for the three default words provided: `['cormulent', 'incendenece', 'validrate']`.

# In[ ]:

from nltk.corpus import words
from nltk.util import ngrams
from nltk.metrics.distance import jaccard_distance

correct_spellings = words.words()


# ### Question 9
# 
# For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:
# 
# **[Jaccard distance](https://en.wikipedia.org/wiki/Jaccard_index) on the trigrams of the two words.**
# 
# *This function should return a list of length three:
# `['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*

# In[ ]:

def answer_nine(entries=['cormulent', 'incendenece', 'validrate']):

    rec = []
    for e in entries:
        rec_subs = [w for w in correct_spellings if w.startswith(e[0])]
        dists = []
        for w in rec_subs:
            dists.append(jaccard_distance(set(ngrams(w, 3)), set(ngrams(e, 3))))

        rec.append(rec_subs[np.argmin(dists)])
        
    return rec
    
answer_nine()


# ### Question 10
# 
# For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:
# 
# **[Jaccard distance](https://en.wikipedia.org/wiki/Jaccard_index) on the 4-grams of the two words.**
# 
# *This function should return a list of length three:
# `['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*

# In[ ]:

def answer_ten(entries=['cormulent', 'incendenece', 'validrate']):
    
    rec = []
    for e in entries:
        rec_subs = [w for w in correct_spellings if w.startswith(e[0])]
        dists = []
        for w in rec_subs:
            dists.append(jaccard_distance(set(ngrams(w, 4)), set(ngrams(e, 4))))

        rec.append(rec_subs[np.argmin(dists)])
        
    return rec

answer_ten()


# ### Question 11
# 
# For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:
# 
# **[Edit distance on the two words with transpositions.](https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance)**
# 
# *This function should return a list of length three:
# `['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*

# In[ ]:

from nltk.metrics.distance import edit_distance
def answer_eleven(entries=['cormulent', 'incendenece', 'validrate']):
    
    rec = []
    for e in entries:
        rec_subs = [w for w in correct_spellings if w.startswith(e[0])]
        dists = []
        for w in rec_subs:
            dists.append(edit_distance(w, e, transpositions=True))

        rec.append(rec_subs[np.argmin(dists)])
        
    return rec
    
answer_eleven()

