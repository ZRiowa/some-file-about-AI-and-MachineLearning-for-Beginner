
# coding: utf-8

# In[1]:


import pandas as pd

# some lines have too many fields (?), so skip bad lines
imgatt = pd.read_csv("image_attribute_labels.txt",sep='\s+',header=None, error_bad_lines=False, 
                    warn_bad_lines=False,usecols=[0,1,2], names=['imgid','attid','present'])

# description from dataset README:
#
# The set of attribute labels as perceived by MTurkers for each image
# is contained in the file attributes/image_attribute_labels.txt, with
# each line coresponding to one image/attribute/worker triplet:
#
# <image_id> <attribute_id> <is_present> <certainty_id> <time>
#
# where <image_id>, <attributes_id>, <certainty_id> corespond to the IDs
#in images.txt, attributes/attributes.txt, and attributes/certainties.txt
#


# In[2]:


imgatt.head()


# In[3]:


imgatt.shape


# In[4]:


# need to reorganize imgatt to have one row per imgid,and 312 columns ( one column per attribute),
# with 1/0 in each cell representing uf that imgid has that attribute or not 

imgatt2 = imgatt.pivot(index = 'imgid', columns = 'attid', values = 'present')


# In[5]:


imgatt2.head()


# In[6]:


imgatt2.shape


# In[7]:


# now we need to load the image true classes

imglabels = pd.read_csv("image_class_labels.txt",
                        sep=' ', header=None, usecols=[0,1], names=['imgid', 'label'])

imglabels = imglabels.set_index('imgid')

# description from dataset README:
# The ground truth class labels (bird species labels) for each image are contained
# in the file image_class_labels.txt, with each line corresponding to one image:
#
# <image_id> <class_id>
#
# where <image_id> and <class_id> correspond to IDs in images.txt and classes.txt,
# respectively.


# In[8]:


imglabels.head()


# In[9]:


imglabels.shape


# In[10]:


# now we need to attach the labels to the attribute data set,
# and shuffle; then we'll separate a test set from a training set

df = imgatt2.join(imglabels)
df = df.sample(frac=1)


# In[11]:


df.head()


# In[12]:


df_att = df.iloc[:, :312]
df_label = df.iloc[:,312:]


# In[13]:


df_att.head()


# In[14]:


df_label.head()


# In[15]:


df_train_att = df_att[:8000]
df_train_label = df_label[:8000]
df_test_att = df_att[8000:]
df_test_label = df_label[8000:]

df_train_label = df_train_label['label']
df_test_label = df_test_label['label']


# In[16]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_features=50, random_state=0, n_estimators=100)


# In[17]:


clf.fit(df_train_att, df_train_label)


# In[18]:


print(clf.predict(df_train_att.head()))


# In[19]:


clf.score(df_test_att, df_test_label)


# In[20]:


from sklearn.metrics import confusion_matrix
pred_labels = clf.predict(df_test_att)
cm = confusion_matrix(df_test_label, pred_labels)


# In[21]:


cm


# In[22]:


# from http://scikit-learn.org/stable/auto_example.model_selection/plot_confusionJ_matrix.html
import matplotlib.pyplot as plt
import itertools
def plot_confusion_matrix(cm, classes,
                         normalize=False,
                         title='Confusion Matrix',
                         cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization!")
        
    print(cm)
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    
    fmt= '0.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #     plt.text(j, i, format(cm[i, j], fmt)
    #              horizontalalignment="center",
    #              color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[23]:


birds = pd.read_csv("classes.txt",
                   sep='\s+', header=None, usecols=[1],names=['birdname'])
birds = birds['birdname']


# In[24]:


birds


# In[25]:


import numpy as np
np.set_printoptions(precision=2)
plt.figure(figsize=(60, 60), dpi=300)
plot_confusion_matrix(cm, classes=birds, normalize=True)
plt.show()


# In[29]:


from sklearn import tree
clftree = tree.DecisionTreeClassifier()
clftree.fit(df_train_att, df_train_label)
clftree.score(df_test_att, df_test_label)


# In[30]:


from sklearn import svm
clfsvm = svm.SVC()
clfsvm.fit(df_train_att, df_train_label)
clfsvm.score(df_test_att, df_test_label)


# In[31]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, df_train_att, df_train_label, cv=5)
print(scores)


# In[32]:


# show average score and +/- two standard deviations away (covering 95% of scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# In[34]:


scorestree = cross_val_score(clftree, df_train_att, df_train_label, cv=5 )
print("Accuracy: %0.2f (+/- %0.2f)" % (scorestree.mean(), scorestree.std() * 2))


# In[37]:


scoressvm = cross_val_score(clfsvm, df_train_att, df_train_label, cv=5)
print("Accuracy: %0.2f (+/- %0.2f) " % (scoressvm.mean(), scoressvm.std() * 2 ))


# In[38]:


max_features_opts = range(5, 50, 5)
n_estimators_opts = range(10, 200, 20)
rf_params = np.empty((len(max_features_opts)*len(n_estimators_opts),4), float)
i =0
for max_features in max_features_opts:
    for n_estimators in n_estimators_opts:
        clf = RandomForestClassifier(max_features=max_features, n_estimators=n_estimators)
        scores = cross_val_score(clf, df_train_att, df_train_label, cv=5)
        rf_params[i, 0] = max_features
        rf_params[i, 1] = n_estimators
        rf_params[i, 2] = scores.mean()
        rf_params[i, 3] = scores.std() * 2
        i+=1
        print("Max features: %d, num estimators: %d ,Accuracy: %0.2f (+/- %0.2f)" % 
             (max_features, n_estimators, scores.mean(), scores.std() * 2))


# In[42]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
fig = plt.figure()
fig.clf()
ax = fig.gca(projection="3d")
x = rf_params[:, 0]
y = rf_params[:, 1]
z = rf_params[:, 2]
ax.scatter(x, y, z)
ax.set_zlim(0.2, 0.5)
ax.set_xlabel("Max features")
ax.set_ylabel("Num sctimators")
ax.set_zlabel("Average Accuracy")
plt.show()

