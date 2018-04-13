
# coding: utf-8

# In[5]:


import gensim, logging  #logging => status messages 状态信息 


# In[6]:


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# In[7]:


gmodel = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary = True)


# In[9]:


gmodel['cat']


# In[10]:


gmodel['dog']


# In[11]:


gmodel


# In[12]:


gmodel.similarity('cat','dog')


# In[13]:


gmodel.similarity('cat', 'spatula')


# In[14]:


from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec


# In[15]:


#提取单词，剔除无用符合，提取出的形成一个list
def extract_words(sent):
    sent = sent.lower()
    sent = re.sub(r'<[^>]+>', ' ', sent)   # strip html tags   
    sent = re.sub(r'(\w)\'(\w)', '\1\2', sent) # remove apostraphes 省略号 '   简写使用
    sent = re.sub(r'\W', ' ', sent)   #remove punctuation
    sent = re.sub(r'\s+', ' ', sent)  #remove repeated space
    return sent.split()


# In[16]:


# unsupervised training data
import re
import os
unsup_sentences = []
for dirname in ["train/pos", "train/neg", "train/unsup", "test/pos", "test/neg"]:
    for fname in sorted(os.listdir("C:/Users/ZooRio/Downloads/Compressed/aclImdb_v1/aclImdb/" + dirname)):
        if fname[-4:] == ".txt" :
            with open ("C:/Users/ZooRio/Downloads/Compressed/aclImdb_v1/aclImdb/" + dirname + "/" + fname, encoding = 'UTF-8') as f:
                sent = f.read()
                words = extract_words(sent)
                unsup_sentences.append(TaggedDocument(words, [dirname + "/" + fname]))


# In[28]:


for dirname in ["neg", "pos"]:
    for fname in sorted(os.listdir("C:/Users/ZooRio/Downloads/Compressed/mix20_rand700_tokens_cleaned/tokens/" + dirname)):
        if fname[-4:] == ".txt" :
            with open ("C:/Users/ZooRio/Downloads/Compressed/mix20_rand700_tokens_cleaned/tokens/" + dirname + "/" + fname, encoding = 'UTF-8') as f:
                for i, sent in enumerate(f):        
                    words = extract_words(sent)
                    unsup_sentences.append(TaggedDocument(words, ["%s-%s-%d" % (dirname, fname, i)]))
#这个步骤的数据出问题，无法处理，但是过程可以参考


# In[27]:


with open ("C:/Users/ZooRio/Downloads/Compressed/stanfordSentimentTreebank/stanfordSentimentTreebank/original_rt_snippets.txt", encoding = 'UTF-8') as f:
    for i,line in enumerate(f):
        words = extract_words(line)
        unsup_sentences.append(TaggedDocument(words, ["rt-%d" % i]))

#对于一个可迭代的（iterable）/可遍历的对象（如列表、字符串），enumerate将其组成一个索引序列，利用它可以同时获得索引和值
#(0, seq[0]), (1, seq[1]), (2, seq[2])


# In[29]:


len(unsup_sentences)


# In[30]:


unsup_sentences[:10]


# In[32]:


import random
class PermuteSentences(object):         
    def __init__(self, sents):
        self.sents = sents
    
    def __iter__(self):
        shuffled = list(self.sents)
        random.shuffle(shuffled)          #random.shuffle会直接作用于原始数列上，直接将原始数列变成乱序
        for sent in shuffled:
            yield sent
            
# permute:交换，改变顺序，置换，排列


# In[35]:


permuter = PermuteSentences(unsup_sentences)
model = Doc2Vec(permuter, dm=0, hs=1, size=50)


# In[36]:


# done with training, free up some memory
model.delete_temporary_training_data(keep_inference=True)


# In[37]:


model.save('reviews.d2v')
# in other program, we could write: model = Doc2Vec.load('reviews.d2v')


# In[38]:


model.infer_vector(extract_words("This place is not worth your time, let alone Vegas."))


# In[41]:


from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(
    [model.infer_vector(extract_words("This place is not worth your time, let alone Vegas."))],
    [model.infer_vector(extract_words("Services sucks."))])


# In[53]:


sentences = []
sentvecs = []
sentiments = []
for fname in ["amazon_cells", "imdb", "yelp"]:
    with open ("C:/Users/ZooRio/Downloads/Compressed/sentiment labelled sentences/sentiment labelled sentences/%s_labelled.txt" % (fname), encoding = 'UTF-8') as f:
        for i,line in enumerate(f):                   #\n 意味着重启一行，也同时意味着不同实例，使用python的split（）函数会自动分割
            line_split = line.strip().split('\t')     #strip() 用来去掉string的前后空格等whitespace类型符号
            sentences.append(line_split[0])           #\t \n 在txt文本中不一定能显示出来，需要通过读取才知道
            words = extract_words(line_split[0])
            sentvecs.append(model.infer_vector(words, steps=10))    #create a vector for this document
            sentiments.append(int(line_split[1]))
            
# shuffle sentences, sentvecs, sentiments together
combined = list(zip(sentences, sentvecs, sentiments))   #先压缩捆绑到一起，再整体打乱顺序，保证关系一一对应
random.shuffle(combined)
sentences, sentvecs, sentiments = zip(*combined)     #解开捆绑，方便接下来的使用


# In[73]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

kclf = KNeighborsClassifier(n_neighbors=9)
clfrf = RandomForestClassifier(n_estimators=100)


# In[74]:


scores = cross_val_score(kclf, sentvecs, sentiments, cv=5)
np.mean(scores), np.std(scores)


# In[58]:


scores = cross_val_score(clfrf, sentvecs, sentiments, cv=5)
np.mean(scores), np.std(scores)


# In[68]:


# bag-of-words comparison
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
pipeline = make_pipeline(CountVectorizer(), TfidfTransformer(), RandomForestClassifier(n_estimators =100))
pipeline


# In[77]:


scores = cross_val_score(pipeline, sentences, sentiments, cv=5)


# In[78]:


np.mean(scores), np.std(scores)

