
# coding: utf-8

# In[1]:


# PCA 


# In[2]:


# get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

# plt.rcParams['figure.figsize'] = [10, 5]


# In[3]:


names = ['class','alcohol','malic_acid','ash','alcalinity_of_ash','magnesium','total_phenols','flavanoids','nonflavanoid_phenols'
          ,'proanthocyanins','color_intensity','hue','OD280_OD315_of_diluted_wines','proline']
data = pd.read_csv('../Data/wine.data', names=names)


# In[4]:


data.head()


# In[5]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, make_scorer

# In[6]:


X = data.drop('class', axis=1)
y = data.loc[:,'class']


# ### Testing without Standarization

# In[7]:


svm_not_scalled = LinearSVC(C=1, loss="hinge")


# In[8]:


print(cross_val_score(svm_not_scalled, X, y, cv=10))


# ### Testing with Standarization

# In[9]:


svm_scalled = Pipeline((
                    ("scaler", StandardScaler()),
                    ("linear_svc", LinearSVC(C=1, loss="hinge"))
                ))


# In[10]:


print(cross_val_score(svm_scalled, X, y, cv=10))


# ### Creating methods to test

# In[15]:


def metricas(classificador, X, y, folds):
    weighted_recall_scorer = make_scorer(recall_score, average='weighted')
    recall = cross_val_score(classificador, X, y, cv=folds, scoring=weighted_recall_scorer)
#     print('Revocação: ', np.mean(recall), recall)
    weighted_precision_scorer = make_scorer(precision_score, average='weighted')
    precision = cross_val_score(classificador, X, y, cv=folds, scoring=weighted_precision_scorer)
#     print('Precisão: ', np.mean(precision), precision)
    accuracy = cross_val_score(classificador, X, y, cv=folds, scoring='accuracy')
#     print('Acurácia', np.mean(accuracy), accuracy)
    return (np.mean(recall), np.mean(precision), np.mean(accuracy))

def scalling_and_svc(X, y, kernel=["linear"], C=[1], gamma=[1], degree=[3], folds=10):
    result = dict() # cria um dicionário dos resultados, com o índice os kernels
    for k in kernel:
        paramC = dict()
        for i in C:
            if k == "rbf":
                paramG = dict()
                for g in gamma:
                    svm_scalled = Pipeline((
                                ("scaler", StandardScaler()),
                                ("svc", SVC(kernel=k, gamma=g, C=i))
                            ))
                    svm_scalled.fit(X, y)
                    paramG[g] = metricas(svm_scalled, X, y, folds)
                paramC[i] = paramG 
            elif k == "poly":
                paramD = dict()
                for d in degree:
                    svm_scalled = Pipeline((
                                ("scaler", StandardScaler()),
                                ("svc", SVC(kernel=k, C=i, coef0=1, degree=d))
                            ))
                    svm_scalled.fit(X,y)
                    paramD[d] = metricas(svm_scalled, X, y, folds)
                paramC[i] = paramD[d]
            elif k == "linear": #case linear
                svm_scalled = Pipeline((
                                ("scaler", StandardScaler()),
                                ("svc", SVC(kernel=k, C=i))
                            ))
                svm_scalled.fit(X,y)
                paramC[i] = metricas(svm_scalled, X, y, folds)
        result[k] = paramC
    return result
    
kernel = ["linear", "poly", "rbf"]
pC = [0.001 , 0.01, 0.1, 1, 10, 100, 1000]
pGamma = [0.01, 0.1, 1, 10, 100]
degree=[3]
folds = 10

# scalling_and_svc(X=X, y=y, kernel = kernel, C=C )
resultados = scalling_and_svc(X, y, kernel, pC, degree, pGamma, folds)
print(resultados)