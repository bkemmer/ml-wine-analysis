
# coding: utf-8

# # Código para fazer uma validação cruzada estratificada

# In[12]:


import numpy as np
import pandas as pd


# In[2]:


def CrossValidacaoEstratificada(dataset, y, folds=4, seed=42):
    np.random.seed(seed)
    npY = np.array(y)
    fold_classes = list()

    #construindo as estruturas
    dataset_fold = list()
    for i in range(folds):
        dataset_fold.append(list())
    fold_atual = 0
    unicos = np.unique(npY)
    for i in range(len(unicos)):
        # cria uma lista das classe e os valores os índices(posições) delas no vetor y
        fold_classes.append(np.where(npY == unicos[i])[0].tolist())
        
        while len(fold_classes[i])>0:
            # sorteia um elemento do vetor de elementos da mesma classe
            if (fold_atual >= folds):
                fold_atual = 0
            index_elemento = np.random.randint(len(fold_classes[i]))
            index = fold_classes[i].pop(index_elemento)
            # Adiciona o elemento sorteado no bucket correspondente
            dataset_fold[fold_atual].append(index)
            fold_atual = fold_atual + 1
    for i in range(len(dataset_fold)):
        print("dataset_fold[" + str(i) + "]: " + str(len(dataset_fold[i])))
    return dataset_fold


names = ['class','alcohol','malic_acid','ash','alcalinity_of_ash','magnesium','total_phenols','flavanoids','nonflavanoid_phenols'
          ,'proanthocyanins','color_intensity','hue','OD280_OD315_of_diluted_wines','proline']
data = pd.read_csv('../Data/wine.data', names=names)


# In[14]:


data_binario = data.loc[data["class"] != 3,:]
data_binario["class"].unique()


# In[15]:


X = data_binario.drop('class', axis=1)
y = data_binario.loc[:,'class']


# In[16]:


dataset_indexes = CrossValidacaoEstratificada(X, y, folds=4)

