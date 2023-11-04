-- naive_bayesian_classsifier
-- using trin.csv file

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('train.csv')
df.head()

df.ndim

df.shape

df.columns

df.info()

df.isna().sum()

df.isna().any()

df.describe().transpose()

null_df=pd.DataFrame()
null_df['Features']=df.isnull().sum().index
null_df['Null values']=df.isnull().sum().values
null_df['% Null values']=(df.isnull().sum().values / df.shape[0])*100
null_df.sort_values(by='% Null values',ascending=False)

import missingno as no
no.bar(df,color='lightblue')
plt.show()

df['Age'].fillna(df.Age.median(),inplace=True)
df.Age.isna().any()

df.drop(columns='Cabin' , inplace=True)

df.dropna(subset=['Embarked'],inplace=True)

df.shape

df.isna().sum()

no.bar(df,color='pink')

"""### Data Preprocessing"""

import  yellowbrick
from sklearn.preprocessing import LabelEncoder
label_Enc =LabelEncoder()

df.Sex =label_Enc.fit_transform(df.Sex)

label_Enc.classes_

df.head()

fig,ax=plt.subplots(ncols=2,figsize=(20,8))
resign_corr = df.corr()
mask = np.triu(np.ones_like(resign_corr, dtype=np.bool))
cat_heatmap = sns.heatmap(df.corr(), mask=mask, vmin=-1, vmax=1,annot=True,ax=ax[0],cmap='Pastel1')
cat_heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':14}, pad=12);
heatmap = sns.heatmap(resign_corr[['Survived']].sort_values(by='Survived',ascending=False),vmin=-1, vmax=1, annot=True,ax=ax[1],cmap='Pastel1')
heatmap.set_title('Features Correlating with Resignation', fontdict={'fontsize':16}, pad=16);

plt.figure(figsize=(9,7))
sns.heatmap(df.corr(),annot=True,cmap='Pastel1')
plt.show()

sns.clustermap(df.corr(),annot=True,cmap='Pastel1')

"""### Dataset Splitting"""

X=df.drop(['Survived','Name','Ticket','Embarked'],axis=1)
y=df['Survived']

X[:3]

y[:5]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state =42)

"""### Gaussian Naive Bayes Calssifier"""

from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
from sklearn.metrics import recall_score, precision_score, f1_score

from sklearn.naive_bayes import GaussianNB

gnb_clf=GaussianNB()

gnb_clf.fit(X_train, y_train)

y_pred = gnb_clf.predict(X_test)

print("Accuracy Score :",accuracy_score(y_test,y_pred))

print("Recall Score",recall_score(y_test,y_pred))

print("Precision Score :",precision_score(y_test,y_pred))

print("F1 Score :",f1_score(y_test,y_pred))

"""### Confusion Matrix"""

confusion_matrix(y_test, y_pred)

print(classification_report(y_test,y_pred))

import yellowbrick as yb
plt.figure(figsize=(15,7))
visualizer = yb.classifier.classification_report(gnb_clf, X_train, y_train, X_test, y_test,  support=True,cmap="BuGn")
visualizer.show()
plt.show()

