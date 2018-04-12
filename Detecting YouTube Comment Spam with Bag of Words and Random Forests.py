
# coding: utf-8

# In[2]:


import pandas as pd
d = pd.read_csv("Youtube01-Psy.csv")


# In[3]:


d.tail()


# In[4]:


len(d.query('CLASS ==1'
           ))


# In[9]:


len(d.query('CLASS == 0'))


# In[12]:


d.query





# In[13]:


len(d)


# In[14]:


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()


# In[15]:


dvec = vectorizer.fit_transform(d['CONTENT'])


# In[16]:


dvec


# In[17]:


analyze = vectorizer.build_analyzer()


# In[18]:


print(d['CONTENT'][349
                  ])
analyze(d['CONTENT'][349])


# In[20]:


vectorizer.get_feature_names()


# In[21]:


dshuf = d.sample(frac=1)    # fration 代表比例  frac=1   表示 100%      


# In[22]:


d_train = dshuf[:300]
d_test = dshuf[300:]
d_train_att = vectorizer.fit_transform(d_train['CONTENT']) #fit bag-of-words on training set
d_test_att = vectorizer.transform(d_test['CONTENT'])   #reuse on testing set
d_train_label = d_train['CLASS']
d_test_label = d_test['CLASS']



# In[31]:


d_train_label  #shuffle the sequences


# In[30]:


d_train_att


# In[28]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators = 80)


# In[29]:


clf.fit(d_train_att,d_train_label)


# In[32]:


clf.score(d_test_att,d_test_label)


# In[34]:


from sklearn.metrics import confusion_matrix
pred_labels = clf.predict(d_test_att)
confusion_matrix(d_test_label,pred_labels)


# In[52]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf , d_train_att, d_train_label, cv =5)
#show average score and +/- two standard deviations away (covering 95% of scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# In[36]:


# load all datasets and combine them
d = pd.concat([pd.read_csv("Youtube01-Psy.csv"),
               pd.read_csv("Youtube02-KatyPerry.csv"),
               pd.read_csv("Youtube03-LMFAO.csv"),
               pd.read_csv("Youtube04-Eminem.csv"),
               pd.read_csv("Youtube05-Shakira.csv")])


# In[37]:


len(d)


# In[38]:


len(d.query('CLASS ==1'))


# In[39]:


len(d.query('CLASS == 0'))


# In[41]:


dshuf = d.sample(frac=1)
d_content = dshuf['CONTENT']
d_label = dshuf['CLASS']


# In[43]:


# set up a pipeline
from sklearn.pipeline import Pipeline, make_pipeline
pipeline = Pipeline([
    ('bag-of-words', CountVectorizer()),    #给每个过程取名
    ('random forest', RandomForestClassifier()),
])
pipeline


# In[44]:


# or: pipeline = make_pipeline(CountVectorizer(), RandomForestClassifier())
make_pipeline(CountVectorizer(), RandomForestClassifier())         


# In[45]:


pipeline.fit(d_content[:1500],d_label[:1500])      #fit the training set,训练


# In[46]:


pipeline.score(d_content[:1500],d_label[:1500])      #score of train sets


# In[47]:


pipeline.predict(["what a neat video!"]) 


# In[48]:


pipeline.predict(["plz subscribe to my channel!"])   


# In[51]:


scores = cross_val_score(pipeline, d_content, d_label, cv = 5)     # 交叉验证   K-FOLD CROSS VALIDATION
print("Accuracy: %0.2f (+/- %0.2f)" % ( scores.mean(), scores.std() * 2))


# In[54]:


pipeline.score(d_content[1500:], d_label[1500:])     #score of test sets


# In[61]:


# add tfidf == TF-IDF   Term-Frequency  |  Inverse-Documents-Frequency   词频
from sklearn.feature_extraction.text import TfidfTransformer
pipeline2 = make_pipeline(CountVectorizer(),
                          TfidfTransformer(norm=None),      
                          RandomForestClassifier(n_estimators = 80))


# In[62]:


scores = cross_val_score(pipeline2, d_content, d_label, cv =5 )
print("Accuracy: %0.2f (+/-%0.2f)" % (scores.mean(), scores.std() * 2))


# In[63]:


pipeline2.steps


# In[67]:


# parameter search    
parameters = {
    'countvectorizer__max_features': (None, 1000, 2000),   
    'countvectorizer__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    'countvectorizer__stop_words': ('english', None),    #give up common words   "english"是个词典
    'tfidftransformer__use_idf': (True, False),  # effectively turn on/off tfidf
    'randomforestclassifier__n_estimators': (20, 50, 100)
}
from sklearn.model_selection import GridSearchCV    # GridSearchCV 给出各种参数下的运行结果
grid_search = GridSearchCV(pipeline2, parameters, n_jobs= -1, verbose= 1)    
#n_jobs 代表使用几个CPU核心，-1 表示马力全开，1 == 1个核心     verbose 表示显示详细信息


# In[68]:


grid_search.fit(d_content, d_label)


# In[69]:


print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r " % (param_name, best_parameters[param_name]))    #打印最佳参数


# In[ ]:




