
# coding: utf-8

# In[1]:
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
import re, string
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier,  GradientBoostingClassifier
import math
%pylab inline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.svm import SVC
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef, classification_report
import nltk
nltk.download('stopwords')
from pymystem3 import Mystem
import matplotlib.cm as cm
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from collections import defaultdict
from gensim.models.word2vec import Word2Vec
from sklearn.multioutput import MultiOutputClassifier
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
import itertools

def sent_count(text):
    pattern = re.compile(r'([А-ЯA-Z]((т.п.|т.д.|пр.|г.)|[^?!.\(]|\([^\)]*\))*[.?!])')
    n = len(pattern.findall(text))     
    return n


# In[2]:


def news_to_wordlist(news, lemmat =False, remove_stopwords=True):
    m = Mystem()
    # 1. Remove non-letters
    news_text= re.sub(r'\W|\b[https]+\b|\b\w{1}\b', ' ', news)
    # 2. lemmatize strings
    if lemmat:
        words = m.lemmatize(news_text)
        words = ''.join(words).strip('\n')
        words = words.split()
        
    # 3. Convert words to lower case and split them
    else:
        words = news_text.lower().split()
    
    # 4. Optionally remove stop words
    if remove_stopwords:
        stops = set(nltk.corpus.stopwords.words("russian"))
        words = [w for w in words if not w in stops]
    # 5. Return a list of words
    return(words)


# In[4]:


def test_split(data, y):
    train_share = int(.7 * data.shape[0])
    x_train_split, y_train_split = data[:train_share], y[:train_share]
    x_test_split, y_test_split  = data[train_share:], y[train_share:]
    return x_train_split, y_train_split, x_test_split, y_test_split



# In[7]:


def score_test(model, x_test, target_test):
    pred_values = model.predict(x_test)
    pres = precision_score(target_test, pred_values)
    rec= recall_score(target_test, pred_values)
    f1 = f1_score(target_test, pred_values)
    accur = accuracy_score(target_test, pred_values)
    conf_matrix_test = confusion_matrix(target_test, pred_values)
    conf_matrix_test_norm = conf_matrix_test.astype('float') / conf_matrix_test.sum(axis=1)[:, np.newaxis]
    return pres, rec, f1, accur, conf_matrix_test, conf_matrix_test_norm


# In[8]:


def score_multi_test(target_test, pred_values):
    pres = precision_score(target_test, pred_values)
    rec= recall_score(target_test, pred_values)
    f1 = f1_score(target_test, pred_values)
    accur = accuracy_score(target_test, pred_values)
    conf_matrix_test = confusion_matrix(target_test, pred_values)
    conf_matrix_test_norm = conf_matrix_test.astype('float') / conf_matrix_test.sum(axis=1)[:, np.newaxis]
    return pres, rec, f1, accur, conf_matrix_test, conf_matrix_test_norm


# In[9]:


def select_threshold(y_true_train, y_pred_train, y_pred_test):
    
    for i in range(y_pred_train.shape[1]):
        list_f1= []
        threshold = (np.array(sorted(y_pred_train[:, i]))[:-1] + np.array(sorted(y_pred_train[:, i]))[1:])/2
        for thres in threshold:
            list_n = np.zeros(len(y_pred_train[:, i]))
            list_n[y_pred_train[:, i]>thres] = 1
            list_f1.append(f1_score(y_true_train.iloc[:, i], list_n))
        if sum(list_f1) == 0.0:
            z = threshold[-1]
        else:
            z = threshold[argmax(list_f1)]
        y_pred_test[:, i][y_pred_test[:, i]>z] =1
        y_pred_test[:, i][y_pred_test[:, i]<z] =0
    y_pred_test = pd.DataFrame(y_pred_test, columns = y_true_train.columns)
    return y_pred_test


# In[10]:


def score_test_multi(target_test, pred_values):
    pres = precision_score(target_test, pred_values)
    rec= recall_score(target_test, pred_values)
    f1 = f1_score(target_test, pred_values)
    accur = accuracy_score(target_test, pred_values)
    conf_matrix_test = confusion_matrix(target_test, pred_values)
    conf_matrix_test_norm = conf_matrix_test.astype('float') / conf_matrix_test.sum(axis=1)[:, np.newaxis]
    return pres, rec, f1, accur, conf_matrix_test, conf_matrix_test_norm

