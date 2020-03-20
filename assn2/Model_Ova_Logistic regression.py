#!/usr/bin/env python
# coding: utf-8

# In[60]:


import utils
import predict
import time as tm
import numpy as np
import scipy
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

k = 5
dictSize = 225
(X, y) = utils.loadData( "C:\\Users\\asus\\Documents\\GitHub\\ml19-20w\\assn2\\train", dictSize = dictSize )
clf = LogisticRegression(random_state=0, solver ='liblinear', multi_class='ovr')
tic = tm.perf_counter()
model = clf.fit(X,y)
prob = model.predict_proba(X[0]) #checking for X[0]
k_pred = np.argsort(prob,axis=1)[:,-k :] # taking top 5 predictions
class_labels= clf.classes_
toc = tm.perf_counter()
print(class_labels[k_pred], toc-tic)
#model = clf.fit(X[0:8000], y[0:8000])
#model.score(X[8000:10000],y[8000:10000])



# In[57]:


kf = KFold(n_splits=10)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model = clf.fit(X_train,y_train)
    print(model.score(X_test,y_test))


# In[ ]:




