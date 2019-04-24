#!/usr/bin/env python
# coding: utf-8

# In[69]:


import pandas as pd # provide sql-like data manipulation tools. very handy.
pd.options.mode.chained_assignment = None
import numpy as np # high dimensional vector computing library.
from copy import deepcopy
from cleaner import *
import re
import string
from string import punctuation
from random import shuffle
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
import time
from IPython.display import display
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.feature_selection import chi2
import numpy as np
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')
import gensim
from gensim.models.word2vec import Word2Vec 
LabeledSentence = gensim.models.doc2vec.LabeledSentence 
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk.
tokenizer = TweetTokenizer()

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


# In[70]:


df = pd.read_csv('data/final_data.csv')
df.shape


# In[71]:


df = df[pd.notnull(df['Tweet'])]


# In[72]:


col = ['Category', 'Tweet']
df = df[col]


# In[73]:


df.columns = ['Category', 'Tweet']


# In[74]:


def clean_text_round1(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


# In[75]:


def clean_text_round2(text):
    '''Get rid of some additional punctuation and non-sensical 
    text that was missed the first time around.'''
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', '', text)
    return text


# In[76]:


from generator import*
data_xml=[]

stop_words = ['ourselves', 'hers', 'between', 'yourself', 
              'but', 'again', 'there', 'about', 'once', 
              'during', 'out', 'very', 'with', 'they',
              'own', 'an', 'be', 'some', 'for', 'do', 'its', 
              'yours', 'such', 'into', 'of', 'most', 'itself',
              'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 
              'from', 'him', 'each', 'the', 'themselves', 'until', 
              'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don',
              'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 
              'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 
              'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 
              'been', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because',
              'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you',
              'herself', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after',
              'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 
              'how', 'further', 'was', 'here', 'than']

def tokenize(tweet):
    tweet = nlp(tweet)
    tweet = negation_tag(tweet)
    tweet = ' '.join(tweet)
    tweet=clean_text_round1(tweet)
    tweet=clean_text_round2(tweet)
    tokens = tokenizer.tokenize(tweet.lower())
    tokens=[w for w in tokens if not w in stop_words]
    
    try:
        temp=[]
        tokens = list(filter(lambda t: not t.startswith('@'), tokens))
        tokens = list(filter(lambda t: not t.startswith('#'), tokens))
        tokens = list(filter(lambda t: not t.startswith('http'), tokens))
        tokens = list(filter(lambda t: not t.startswith('www'), tokens))
        temp.append(tokens)
        #print(type(data.Category.values))
        #temp.append(data.Category)
        data_xml.append(temp)
        

        
        #doc=convert_to_doc(' '.join(tokens))
        #data_xml.append(doc)
        #vector=generator_vec(doc)
        return tokens
    except:
        return 'NC'


# In[77]:


def postprocess(data, n=93170):
    data = data.head(n)
    data['tokens'] = data['Tweet'].progress_map(tokenize) 
    data = data[data.tokens != 'NC']
    data.reset_index(inplace=True)
    return data

data = postprocess(df)


# In[78]:


x_train, x_test, y_train, y_test = train_test_split(np.array(data.head(93170).tokens),
                                   np.array(data.head(93170).Tweet), test_size=0.2)


# In[79]:


def labelizeTweets(tweets, label_type):
    labelized = []
    for i,v in tqdm(enumerate(tweets)):
        label = '%s_%s'%(label_type,i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized

x_train = labelizeTweets(x_train, 'TRAIN')
x_test = labelizeTweets(x_test, 'TEST')


# In[80]:


tweet_w2v = Word2Vec(size=150, min_count=20,sg=0)#sg=1: skip-gram, 0: cbow
tweet_w2v.build_vocab([x.words for x in tqdm(x_train)])
tweet_w2v.train([x.words for x in tqdm(x_train)], 
                total_examples=tweet_w2v.corpus_count, epochs=tweet_w2v.iter)


# In[81]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
from sklearn.decomposition import PCA

def plot_words(words):
    word_vecs = [tweet_w2v[word] for word in words]
    pca = PCA(n_components=2)

    columns = ["Component1","Component2"]
    df = pd.DataFrame(pca.fit_transform(word_vecs), columns=columns, index=words)
    def annotate_df(row):  
        ax.annotate(row.name, list(row.values),
                    xytext=(10,-5), 
                    textcoords='offset points',
                    size=12, 
                    color='black')

    ax = df.plot(kind="scatter",x='Component1', y='Component2',)
    _ = df.apply(annotate_df, axis=1)
    plt.savefig('outputcb.jpg')
    


# In[82]:


words=['meningitis','contracted', 
       'contract',
       'get', 'got',
       'getting',
       'have', 'having', 'had', 'has', 'catch', 'caught', 'infected', 'recovered']

plot_words(words)


# In[83]:


print('building tf-idf matrix ...')
vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=0)
matrix = vectorizer.fit_transform([x.words for x in x_train])
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
print('vocab size :', len(tfidf))


# In[84]:


def buildWordVector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += tweet_w2v[word].reshape((1, size)) * tfidf[word]
            count += 1.
        except KeyError: # handling the case where the token is not
                         # in the corpus. useful for testing.
            continue
    if count != 0:
        vec /= count
    return vec


# In[85]:


data['category_id'] = data['Category'].factorize()[0]
from io import StringIO
category_id_df = data[['Category', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'Category']].values)


# In[86]:


labels=data.category_id


# In[88]:


from sklearn.preprocessing import scale
features = np.concatenate([buildWordVector(z, 150) for z in tqdm(map(lambda x: x, data.tokens))])
features = scale(features)


# # Vectorizing the tweet using the embedding model

# In[90]:


start = time.time()

models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MLPClassifier(),
    #MultinomialNB(),
    LogisticRegression(random_state=0),
]
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
print("Execution time =  {}".format(time.time()-start))


# In[91]:


sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
              size=8, jitter=True, edgecolor="gray", linewidth=2)
#plt.show()
plt.savefig('plot/word2Vec/models.jpg')


# In[92]:


cv_df.groupby('model_name').accuracy.mean()


# ## Support Vector Machine

# In[93]:


model = LinearSVC()

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.33, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# In[96]:


from sklearn.metrics import classification_report

conf_mat = confusion_matrix(y_test, y_pred)

target_names=['Infection','Concern','Vaccine','Campaign','News']
print(classification_report(y_test, y_pred, target_names=target_names))


fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=category_id_df.Category.values, yticklabels=category_id_df.Category.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('plot/word2Vec/svm.jpg')
plt.show()


# ## ANN

# In[97]:


model_ann = MLPClassifier()
model_ann.fit(X_train, y_train)
y_pred = model_ann.predict(X_test)


# In[98]:


from sklearn.metrics import classification_report

conf_mat = confusion_matrix(y_test, y_pred)

target_names=['Infection','Concern','Vaccine','Campaign','News']
print(classification_report(y_test, y_pred, target_names=target_names))

fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=category_id_df.Category.values, yticklabels=category_id_df.Category.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('plot/word2Vec/ann.jpg')
plt.show()


# ## Random Forest

# In[99]:


model_rf = RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0)
model_rf.fit(X_train, y_train)
y_pred = model_rf.predict(X_test)


# In[100]:


from sklearn.metrics import classification_report

conf_mat = confusion_matrix(y_test, y_pred)

target_names=['Infection','Concern','Vaccine','Campaign','News']
print(classification_report(y_test, y_pred, target_names=target_names))

fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=category_id_df.Category.values, yticklabels=category_id_df.Category.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('plot/word2Vec/rf.jpg')
plt.show()


# ## Logistic Regression

# In[101]:


model_lr =  LogisticRegression(random_state=0)
model_lr.fit(X_train, y_train)
y_pred = model_lr.predict(X_test)


# In[102]:


from sklearn.metrics import classification_report

conf_mat = confusion_matrix(y_test, y_pred)

target_names=['Infection','Concern','Vaccine','Campaign','News']
print(classification_report(y_test, y_pred, target_names=target_names))

fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=category_id_df.Category.values, yticklabels=category_id_df.Category.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('plot/word2Vec/lr.jpg')
plt.show()


# In[ ]:




