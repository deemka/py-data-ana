
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-text-mining/resources/d9pwm) course resource._
# 
# ---

# # Assignment 3
# 
# In this assignment you will explore text message data and create models to predict if a message is spam or not. 

# In[199]:

import pandas as pd
import numpy as np

spam_data = pd.read_csv('spam.csv')

spam_data['target'] = np.where(spam_data['target']=='spam',1,0)
spam_data.head(10)


# In[200]:

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(spam_data['text'], 
                                                    spam_data['target'], 
                                                    random_state=0)


# ### Question 1
# What percentage of the documents in `spam_data` are spam?
# 
# *This function should return a float, the percent value (i.e. $ratio * 100$).*

# In[3]:

def answer_one():
    return 100. * spam_data['target'].sum() / len(spam_data['target'])


# In[4]:

answer_one()


# ### Question 2
# 
# Fit the training data `X_train` using a Count Vectorizer with default parameters.
# 
# What is the longest token in the vocabulary?
# 
# *This function should return a string.*

# In[121]:

from sklearn.feature_extraction.text import CountVectorizer

def answer_two():
    vect = CountVectorizer().fit(X_train)
    return sorted(zip(vect.get_feature_names(),
                           map(lambda s: len(s), vect.get_feature_names())),
                  key=lambda t: t[1], reverse=True)[0][0]


# In[12]:

answer_two()


# ### Question 3
# 
# Fit and transform the training data `X_train` using a Count Vectorizer with default parameters.
# 
# Next, fit a fit a multinomial Naive Bayes classifier model with smoothing `alpha=0.1`. Find the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[29]:

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score

def answer_three():
    vectorizer = CountVectorizer().fit(X_train)
    clf = MultinomialNB(alpha=0.1).fit(vectorizer.transform(X_train), y_train)
    return roc_auc_score(y_test, clf.predict(vectorizer.transform(X_test)))


# In[32]:

answer_three()


# ### Question 4
# 
# Fit and transform the training data `X_train` using a Tfidf Vectorizer with default parameters.
# 
# What 20 features have the smallest tf-idf and what 20 have the largest tf-idf?
# 
# Put these features in a two series where each series is sorted by tf-idf value and then alphabetically by feature name. The index of the series should be the feature name, and the data should be the tf-idf.
# 
# The series of 20 features with smallest tf-idfs should be sorted smallest tfidf first, the list of 20 features with largest tf-idfs should be sorted largest first. 
# 
# *This function should return a tuple of two series
# `(smallest tf-idfs series, largest tf-idfs series)`.*

# In[140]:

from sklearn.feature_extraction.text import TfidfVectorizer

def answer_four():
    vrizer = TfidfVectorizer().fit(X_train)
    X_vrzd = vrizer.transform(X_train)
    
    # indices of features, sorted by feature value (=dim 0)
    ids = X_vrzd.max(0).toarray()[0].argsort()
    vals = X_vrzd.max(0).toarray()[0]
    vals.sort()
    feat = np.array(vrizer.get_feature_names())
    
    
    return (pd.Series(vals[:20], index=feat[ids[:20]]),
            pd.Series(vals[-20:], index=feat[ids[-20:]]))


# In[141]:

answer_four()


# ### Question 5
# 
# Fit and transform the training data `X_train` using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **3**.
# 
# Then fit a multinomial Naive Bayes classifier model with smoothing `alpha=0.1` and compute the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[73]:

def answer_five():
    vrizer = TfidfVectorizer(min_df=3).fit(X_train)
    X_vrzd = vrizer.transform(X_train)
    clf = MultinomialNB(alpha=0.1).fit(X_vrzd, y_train)
    return roc_auc_score(y_test, clf.predict(vrizer.transform(X_test)))


# In[74]:

answer_five()


# ### Question 6
# 
# What is the average length of documents (number of characters) for not spam and spam documents?
# 
# *This function should return a tuple (average length not spam, average length spam).*

# In[177]:

def answer_six():
    spam_data['len'] = spam_data['text'].apply(len)
    g = spam_data.groupby('target')['len'].agg(np.mean)
    # spam_data.drop('len', axis=1, inplace=True)
    return (g[0], g[1])


# In[179]:

answer_six()


# <br>
# <br>
# The following function has been provided to help you combine new features into the training data:

# In[46]:

def add_feature(X, feature_to_add):
    """
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    """
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')


# ### Question 7
# 
# Fit and transform the training data X_train using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **5**.
# 
# Using this document-term matrix and an additional feature, **the length of document (number of characters)**, fit a Support Vector Classification model with regularization `C=10000`. Then compute the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[95]:

from sklearn.svm import SVC
def answer_seven():
    vrizer = TfidfVectorizer(min_df=5).fit(X_train)
    X_vrzd = vrizer.transform(X_train)
    X_vrzd = add_feature(X_vrzd, spam_data.iloc[X_train.index].len)
    
    clf = SVC(C=1e4).fit(X_vrzd, y_train)
    X_vrzd_test = add_feature(vrizer.transform(X_test), 
                              spam_data.iloc[X_test.index].len)
        
    return roc_auc_score(y_test, clf.predict(X_vrzd_test))


# In[96]:

answer_seven()


# ### Question 8
# 
# What is the average number of digits per document for not spam and spam documents?
# 
# *This function should return a tuple (average # digits not spam, average # digits spam).*

# In[178]:

def answer_eight():
    spam_data['ndigits'] = spam_data['text'].apply(lambda s: sum(c.isdigit() for c in s))
    g = spam_data.groupby('target')['ndigits'].agg(np.mean)
    # spam_data.drop('ndigits', axis=1, inplace=True)
    return (g[0], g[1])


# In[180]:

answer_eight()


# ### Question 9
# 
# Fit and transform the training data `X_train` using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **5** and using **word n-grams from n=1 to n=3** (unigrams, bigrams, and trigrams).
# 
# Using this document-term matrix and the following additional features:
# * the length of document (number of characters)
# * **number of digits per document**
# 
# fit a Logistic Regression model with regularization `C=100`. Then compute the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[103]:

from sklearn.linear_model import LogisticRegression

def answer_nine():
    vrizer = TfidfVectorizer(min_df=5, ngram_range=(1,3)).fit(X_train)
    X_vrzd = vrizer.transform(X_train)
    
    X_vrzd = add_feature(X_vrzd, spam_data.iloc[X_train.index].len)
    X_vrzd = add_feature(X_vrzd, spam_data.iloc[X_train.index].ndigits)
    
    clf = LogisticRegression(C=100).fit(X_vrzd, y_train)
    X_vrzd_test = add_feature(
            add_feature(vrizer.transform(X_test), 
                              spam_data.iloc[X_test.index].len),
            spam_data.iloc[X_test.index].ndigits)
        
    return roc_auc_score(y_test, clf.predict(X_vrzd_test))


# In[104]:

answer_nine()


# ### Question 10
# 
# What is the average number of non-word characters (anything other than a letter, digit or underscore) per document for not spam and spam documents?
# 
# *Hint: Use `\w` and `\W` character classes*
# 
# *This function should return a tuple (average # non-word characters not spam, average # non-word characters spam).*

# In[181]:

def answer_ten():
    spam_data['nw'] = spam_data['text'].str.findall(r'\W').apply(len)
    g = spam_data.groupby('target')['nw'].agg(np.mean)
    return (g[0], g[1])


# In[182]:

answer_ten()


# ### Question 11
# 
# Fit and transform the training data X_train using a Count Vectorizer ignoring terms that have a document frequency strictly lower than **5** and using **character n-grams from n=2 to n=5.**
# 
# To tell Count Vectorizer to use character n-grams pass in `analyzer='char_wb'` which creates character n-grams only from text inside word boundaries. This should make the model more robust to spelling mistakes.
# 
# Using this document-term matrix and the following additional features:
# * the length of document (number of characters)
# * number of digits per document
# * **number of non-word characters (anything other than a letter, digit or underscore.)**
# 
# fit a Logistic Regression model with regularization C=100. Then compute the area under the curve (AUC) score using the transformed test data.
# 
# Also **find the 10 smallest and 10 largest coefficients from the model** and return them along with the AUC score in a tuple.
# 
# The list of 10 smallest coefficients should be sorted smallest first, the list of 10 largest coefficients should be sorted largest first.
# 
# The three features that were added to the document term matrix should have the following names should they appear in the list of coefficients:
# ['length_of_doc', 'digit_count', 'non_word_char_count']
# 
# *This function should return a tuple `(AUC score as a float, smallest coefs list, largest coefs list)`.*

# In[201]:

def answer_eleven():
    spam_data['length_of_doc'] = spam_data['text'].apply(len)
    spam_data['digit_count'] = spam_data['text'].apply(lambda s: sum(c.isdigit() for c in s))
    spam_data['non_word_char_count'] = spam_data['text'].str.findall(r'\W').apply(len)

    vrizer = CountVectorizer(min_df=5, 
                             ngram_range=(2,5),
                             analyzer='char_wb').fit(X_train)
    
    X_vrzd = vrizer.transform(X_train)
    
    X_vrzd = add_feature(X_vrzd, spam_data.iloc[X_train.index].length_of_doc)
    X_vrzd = add_feature(X_vrzd, spam_data.iloc[X_train.index].digit_count)
    X_vrzd = add_feature(X_vrzd, spam_data.iloc[X_train.index].non_word_char_count)

    X_test_v = vrizer.transform(X_test)
    
    X_test_v = add_feature(X_test_v, spam_data.iloc[X_test.index].length_of_doc)
    X_test_v = add_feature(X_test_v, spam_data.iloc[X_test.index].digit_count)
    X_test_v = add_feature(X_test_v, spam_data.iloc[X_test.index].non_word_char_count)
           
    clf = LogisticRegression(C=100)
    clf.fit(X_vrzd, y_train)
    
    sc = roc_auc_score(y_test, clf.predict(X_test_v))
    
    coeff_ids = clf.coef_[0].argsort()
    
    feat = np.array(vrizer.get_feature_names())
    feat = np.append(feat, ['length_of_doc', 'digit_count', 'non_word_char_count'])
    feat = list(feat[coeff_ids])
    
    return (sc, feat[:10], feat[:-11:-1])


# In[202]:

answer_eleven()

