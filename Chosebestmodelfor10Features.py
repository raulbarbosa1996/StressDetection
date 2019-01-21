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
dataset = pd.read_csv('D:\\Mestrado\\InteligenciaArtificial\\Projeto\\Modelo2\data.csv',sep=';',encoding='ISO-8859-1',engine='python') 


fields = ['Emotional','Carefreeness..N1.','Positive.mood..N3.','Equanimity..N2.','Self.consciousness..N4.','Emot..robustness..N6.','Cheerfulness..E6.','Competence..C1.','Openn..to.actions..O4.','Extraversion..E.','Openn..to.val..norm...O6.']

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
dfff1['Positive'] = dfff1['Positive.mood..N3.'].apply(lambda x: 1 if x < 0.0 else 3 if x>1.0 else 2)
final2=dfff1['Positive']

dffff= DataFrame(columnEmon,columns=['Equanimity..N2.'])
dffff1 = dffff.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',','.'), errors='coerce'))
dffff1['Equanimity'] = dffff1['Equanimity..N2.'].apply(lambda x: 1 if x < 0.0 else 3 if x>0.6 else 2)
final3=dffff1['Equanimity']

dfffff= DataFrame(columnEmon,columns=['Self.consciousness..N4.'])
dfffff1 = dfffff.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',','.'), errors='coerce'))
dfffff1['Consciousness'] = dfffff1['Self.consciousness..N4.'].apply(lambda x: 1 if x < 0.0 else 3 if x>1.0 else 2)
final4=dfffff1['Consciousness']

dffffff= DataFrame(columnEmon,columns=['Emot..robustness..N6.'])
dffffff1 = dffffff.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',','.'), errors='coerce'))
dffffff1['Robustness'] = dffffff1['Emot..robustness..N6.'].apply(lambda x: 1 if x < 0.0 else 3 if x>1.0 else 2)
final5=dffffff1['Robustness']


top6= DataFrame(columnEmon,columns=['Cheerfulness..E6.'])
top61 = top6.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',','.'), errors='coerce'))
top61['Cheerfulness'] = top61['Cheerfulness..E6.'].apply(lambda x: 1 if x < 1.40 else 3 if x>2.10 else 2)
final6=top61['Cheerfulness']

top7= DataFrame(columnEmon,columns=['Competence..C1.'])
top71 = top7.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',','.'), errors='coerce'))
top71['Competence'] = top71['Competence..C1.'].apply(lambda x: 1 if x < 0.50 else 3 if x>1.40 else 2)
final7=top71['Competence']

top8= DataFrame(columnEmon,columns=['Openn..to.actions..O4.'])
top81 = top8.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',','.'), errors='coerce'))
top81['OpenActions'] = top81['Openn..to.actions..O4.'].apply(lambda x: 1 if x < 1.0 else 3 if x>2.0 else 2)
final8=top81['OpenActions']

top9= DataFrame(columnEmon,columns=['Extraversion..E.'])
top91 = top9.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',','.'), errors='coerce'))
top91['Extraversion'] = top91['Extraversion..E.'].apply(lambda x: 1 if x < 0.50 else 3 if x>0.50 else 2)
final9=top91['Extraversion']

top10= DataFrame(columnEmon,columns=['Openn..to.val..norm...O6.'])
top101 = top10.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',','.'), errors='coerce'))
top101['OpenToVal'] = top101['Openn..to.val..norm...O6.'].apply(lambda x: 1 if x < 0.50 else 3 if x>1.40 else 2)
final10=top101['OpenToVal']





dataset['EmotionalFinal']=final
dataset['Carefreeness']=final1
dataset['Positive']=final2
dataset['Equanimity']=final3
dataset['Consciousness']=final4
dataset['Robustness']=final5
dataset['Cheerfulness']=final6
dataset['Competence']=final7
dataset['OpenActions']=final8
dataset['Extraversion']=final9
dataset['OpenToVal']=final10



remove_others(dataset, {'Carefreeness','Positive','Equanimity','Consciousness','Robustness','EmotionalFinal','Cheerfulness','Competence','OpenActions','Extraversion','OpenToVal'})

'''
# head
print(dataset.head(20))

# descriptions
print(dataset.describe())

# class distribution
print(dataset.groupby('class').size())


	
# histograms
dataset.hist()
plt.show()
'''
# Split-out validation dataset
array = dataset.values
X = array[:, 1:11]
Y = array[:, 0]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
seed = 7
scoring = 'accuracy'



	
# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)


fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()