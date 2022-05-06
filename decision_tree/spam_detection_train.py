import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import string
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree

import sys
import os

# Download from https://www.kaggle.com/onlyshadow/spam-or-ham-7-machine-learning-tools-walk-through
df = pd.read_csv('../datasets/spam.csv', encoding='latin-1')
df = df.loc[:,['v1','v2']]
# df.tail()
d={'spam':1,'ham':0}
df.v1 = list(map(lambda x:d[x],df.v1))

import nltk
# nltk.download("punkt")


import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

class stemmed_tfidf():
    def __init__(self,max_features=5000):
        self.ps = PorterStemmer()
        self.vc = TfidfVectorizer(analyzer='word',#{‘word’, ‘char’}  Whether the feature should be made of word or character n-grams
                             stop_words = 'english',
                             max_features = max_features)
    def tfidf(self,ListStr):
        '''
        return: sklearn.feature_extraction.text.TfidfVectorizer
        '''
        table = self.vc.fit_transform([self.stem_string(s) for s in ListStr])
        return table
    def stem_string(self,s):
        '''
        s:str, e.g. s = "Get strings with string. With. Punctuation?"
        ps: stemmer from nltk module
        return: bag of words.e.g. 'get string with string with punctuat'
        '''    
        s = re.sub(r'[^\w\s]',' ',s)# remove punctuation.
        tokens = word_tokenize(s) # list of words.
        #a = [w for w in tokens if not w in stopwords.words('english')]# remove common no meaning words
        return ' '.join([self.ps.stem(w) for w in tokens])# e.g. 'desks'->'desk'

MAX_DEPTH = 5
MAX_FEATURES = 5000

if __name__ == '__main__':
    
    if len(sys.argv) != 3:
        print("Not enough parameters!")
        exit(1)
    
    MAX_DEPTH = int(sys.argv[1])
    MAX_FEATURES = int(sys.argv[2])
    print(f"Parameter Depth: {MAX_DEPTH}, Features: {MAX_FEATURES}")

    # Construct the training and validate set
    stf = stemmed_tfidf(max_features=MAX_FEATURES)
    feature = stf.tfidf(df.v2) # this will be a sparse matrix of size (n,5000)

    Xtrain, Xtest, ytrain, ytest = train_test_split(feature, df.v1, test_size=0.2, random_state=1)

    Acc = {}
    F1score = {}
    confusion_mat={}
    predictions = {}

    # training algorithm
    # val_scores = []
    # for i in range(2,21):
    #     DT = DecisionTreeClassifier(min_samples_split=i, max_depth=MAX_DEPTH, random_state=1,class_weight='balanced')
    #     scores = cross_val_score(DT, Xtrain, ytrain,scoring='f1')
    #     val_scores.append([np.mean(scores),i])
    # val_scores = np.array(val_scores)
    # print('The best scores happens on:',val_scores[val_scores[:,0]==max(val_scores[:,0]),1:],
    #     ', where F1 =',val_scores[val_scores[:,0]==max(val_scores[:,0]),0])
    SPLIT_NUM = 2
    name = 'DT'
    DT = DecisionTreeClassifier(min_samples_split=SPLIT_NUM, max_depth=MAX_DEPTH, random_state=1, class_weight='balanced')
    DT.fit(Xtrain,ytrain)
    pred = DT.predict(Xtest.toarray())
    F1score[name]= f1_score(ytest,pred)
    Acc[name] = accuracy_score(ytest,pred)
    confusion_mat[name] = confusion_matrix(ytest,pred)
    predictions[name]=pred
    print(name+': Accuracy=%1.3f, F1=%1.3f'%(Acc[name],F1score[name]))

    dot_file = "dt.dot"
    output_file = open(dot_file, 'w')
    dot_data = tree.export_graphviz(
        DT, class_names=['0', '1'], filled=True, rounded=True, out_file=output_file)
