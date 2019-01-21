# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 00:19:04 2018

@author: Raul Barbosa
"""

import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 12:30:06 2018

@author: Raul Barbosa
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas import Series, DataFrame

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')
'exec(%matplotlib inline)'

from sklearn.linear_model import LogisticRegression

import sklearn.linear_model as sk

from typing import Set, Any
def remove_others(df: DataFrame, columns: Set[Any]):
    cols_total: Set[Any] = set(df.columns)
    diff: Set[Any] = cols_total - columns
    df.drop(diff, axis=1, inplace=True)

#get titanic and test csv files and insert into Dataframes
dataset = pd.read_csv('D:\\Mestrado\\InteligenciaArtificial\\Projeto\\Modelo2\\data.csv',sep=';',encoding='ISO-8859-1',engine='python') 


fields = ['Emotional','Carefreeness..N1.','Positive.mood..N3.','Equanimity..N2.','Self.consciousness..N4.','Emot..robustness..N6.']

columnEmon = pd.read_csv('D:\\Mestrado\\InteligenciaArtificial\\Projeto\\Modelo2\data.csv',sep=';',encoding='ISO-8859-1',engine='python',skipinitialspace=True,usecols=fields) 


df = DataFrame(columnEmon,columns=['Emotional'])
df1 = df.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',','.'), errors='coerce'))
df1['EmotionalFinal'] = df1['Emotional'].apply(lambda x: 1 if x < -0.5 else 3 if x>0.5 else 2)
final=df1['EmotionalFinal']


dff= DataFrame(columnEmon,columns=['Carefreeness..N1.'])
dff1 = dff.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',','.'), errors='coerce'))
dff1['Carefreeness'] = dff1['Carefreeness..N1.'].apply(lambda x: 1 if x < -0.5 else 3 if x>0.5 else 2)
final1=dff1['Carefreeness']

dfff= DataFrame(columnEmon,columns=['Positive.mood..N3.'])
dfff1 = dfff.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',','.'), errors='coerce'))
dfff1['Positive'] = dfff1['Positive.mood..N3.'].apply(lambda x: 1 if x < 0.41 else 3 if x>1.31 else 2)
final2=dfff1['Positive']

dffff= DataFrame(columnEmon,columns=['Equanimity..N2.'])
dffff1 = dffff.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',','.'), errors='coerce'))
dffff1['Equanimity'] = dffff1['Equanimity..N2.'].apply(lambda x: 1 if x < 0.0 else 3 if x>0.8 else 2)
final3=dffff1['Equanimity']

dfffff= DataFrame(columnEmon,columns=['Self.consciousness..N4.'])
dfffff1 = dfffff.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',','.'), errors='coerce'))
dfffff1['Consciousness'] = dfffff1['Self.consciousness..N4.'].apply(lambda x: 1 if x < 0.21 else 3 if x>1.20 else 2)
final4=dfffff1['Consciousness']

dffffff= DataFrame(columnEmon,columns=['Emot..robustness..N6.'])
dffffff1 = dffffff.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',','.'), errors='coerce'))
dffffff1['Robustness'] = dffffff1['Emot..robustness..N6.'].apply(lambda x: 1 if x < 0.10 else 3 if x>1.10 else 2)
final5=dffffff1['Robustness']


dataset['EmotionalFinal']=final
dataset['Carefreeness']=final1
dataset['Positive']=final2
dataset['Equanimity']=final3
dataset['Consciousness']=final4
dataset['Robustness']=final5


remove_others(dataset, {'Carefreeness','Positive','Equanimity','Consciousness','Robustness','EmotionalFinal'})



# Split-out validation dataset
array = dataset.values
X = array[:, 1:6]
Y = array[:, 0]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
seed = 7
scoring = 'accuracy'




# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


Submission = pd.DataFrame({"EmotinalFinalTeste":Y_validation, "EmotionalFinalPredi": predictions })
Submission.to_csv('finalKNN.csv', index=False)




