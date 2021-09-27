# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 15:49:58 2020

@author: Irfan Sheikh
"""
#Importing Libraries#
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import os
os.chdir("D:\\Data Science\\Project\\Project Hotel Review Excelr")

#Importing dataset#
df=pd.read_excel("D:\\Data Science\\Project\\Project Hotel Review Excelr\\hotel_reviews.xlsx")

#head
df.head()
#Description
df.describe()
#shape
df.shape
#columns
df.columns
#unique values
df.nunique()

#Count of null values#
count=df.isnull().sum().sort_values(ascending=True)
percentage=((df.isnull().sum()/len(df)*100))
missing_data=pd.concat([count,percentage],axis=1,keys=["Count","Percentage"])

#Rating Count
sns.set_style("darkgrid")
sns.countplot(x="Rating",hue="Rating",data=df)


#Percentage of Rating distribution
print(round(df.Rating.value_counts(normalize=True)*100,2))
round(df.Rating.value_counts(normalize=True)*100,2)
round(df.Rating.value_counts(normalize=True)*100,2).plot(kind="bar",figsize=(50,30),color="green",edgecolor="orange")
plt.xlabel("Ratings",fontsize=45)
plt.ylabel("Percentage",fontsize=45)
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)
plt.title("Rating Percentage",fontsize=60)
plt.show()


#Importing libraries for text preprocessing

import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
from nltk.stem import WordNetLemmatizer 

lemmatizer = WordNetLemmatizer()
from nltk.tokenize import word_tokenize
nltk.download('punkt')
stemmer=PorterStemmer()
import nltk
nltk.download('wordnet')
  
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
STOP_WORDS |= {"nt","hotel","room"}
print(STOP_WORDS)


#Cleaning of texts ie punctuations,special characters,numerical values ,lowering of letters
def text_clean(text):
    text=text.lower()
   
    text=re.sub("\[.*?\]","",text)
    text=re.sub("[%s]" % re.escape(string.punctuation),"",text)
    text=re.sub("\w*\d\w*","",text)
    text=re.sub("\n","",text)
    
    return text


#Applying function to dataset 
cleaned1 = lambda x: text_clean(x)
df["Cleaned_Reviews"]=pd.DataFrame(df.Review.apply(cleaned1))







#Performing lemmatization


Reviews1=df.copy()
Reviews1.drop(["Review","Rating"],axis=1,inplace=True)
Reviews1["Cleaned_Reviews"][6]


corpus=[]



for i in range  (0,len(Reviews1)):
    review=re.sub("[^a-zA-Z]"," ",Reviews1["Cleaned_Reviews"][i])
    
    review=review.split()
    review=[lemmatizer.lemmatize(word) for word in review if not word in STOP_WORDS]
    review=" ".join(review)
    corpus.append(review)
 

corpus[6]

df["Cleaned_Review_Lemmatized"]=corpus


#Polarity and subjectivity#
import textblob
from textblob import TextBlob

df["Polarity"]=df["Cleaned_Review_Lemmatized"].apply(lambda x:TextBlob(x).sentiment.polarity)
df["Subjectivity"]=df["Cleaned_Review_Lemmatized"].apply(lambda x:TextBlob(x).sentiment.subjectivity)

def sentiment(x):
    if x<0:
        return 'negative'
    elif x==0:
        return 'neutral'
    else:
        return 'positive'
    
df['polarity_score']=df['Polarity'].\
   map(lambda x: sentiment(x))

pos = [5,4,3]
neg = [1]
neu=[2]

def sentiment(rating):
  if rating in pos:
    return "positive"
  elif rating in neg:
    return "negative"
  elif rating in neu:
      return "neutral"
 
df['Sentiment'] = df['Rating'].apply(sentiment)



#Model Building#

import pandas as pd
import numpy
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
#Splitting data into test and train#


#Using TF-IDF#
tv = TfidfVectorizer(max_features=40000)
X = tv.fit_transform(df["Cleaned_Review_Lemmatized"]).toarray()
X_feat=pd.DataFrame(X)

Reviews=pd.concat([df["Sentiment"],X_feat],axis=1)
Reviews.shape
Reviews.head(10)
X.head()
X=Reviews.iloc[:,1:40001]
Y=Reviews.iloc[:,0]
Y2=Y.values.reshape(1,-1)
X_train1,X_test1,Y_train1,Y_test1=train_test_split(X,Y,test_size=0.3)
Reviews.shape
Reviews.head(10)


classifier_NB = GaussianNB()
classifier_NB.fit(X_train1, Y_train1)
pred_NB_train=classifier_NB.predict(X_train1)
np.mean(pred_NB_train==Y_train1)
pred_NB_test=classifier_NB.predict(X_test1)
np.mean(pred_NB_test==Y_test1)
#Train Accuracy NB=85.74
#Test Accuracy NB=68.91






classifier_MNB = MultinomialNB()
classifier_MNB.fit(X_train1, Y_train1)
pred_MNB_train=classifier_MNB.predict(X_train1)
np.mean(pred_MNB_train==Y_train1)
pred_MNB_test=classifier_MNB.predict(X_test1)
np.mean(pred_MNB_test==Y_test1)
#Train Accuracy MNB=84.52
#Test Accuracy MNB=83.99



classifier_DT = DecisionTreeClassifier(criterion = "entropy", random_state = 0)
classifier_DT.fit(X_train1,Y_train1)
pred_DT_train=classifier_DT.predict(X_train1)
np.mean(pred_DT_train==Y_train1)
pred_DT_test=classifier_DT.predict(X_test1)
np.mean(pred_DT_test==Y_test1)
#Train Accuracy DT=100
#Test Accuracy DT=85.21


classifier_LR=LogisticRegression()
classifier_LR.fit(X_train1,Y_train1)
pred_LR_train=classifier_LR.predict(X_train1)
np.mean(pred_LR_train==Y_train1)
pred_LR_test=classifier_LR.predict(X_test1)
np.mean(pred_LR_test==Y_test1)
#Train Accuracy LR=94.57
#Test Accuracy LR=92.51




classifier_ADA=AdaBoostClassifier()
classifier_ADA.fit(X_train1,Y_train1)
pred_ADA_train=classifier_ADA.predict(X_train1)
np.mean(pred_ADA_train==Y_train1)
pred_ADA_test=classifier_ADA.predict(X_test1)
np.mean(pred_ADA_test==Y_test1)
#Train Accuracy ADA=91.18
#Test Accuracy LR=90.27



classifier_RF=RandomForestClassifier()
classifier_RF.fit(X_train1,Y_train1)
pred_RF_train=classifier_RF.predict(X_train1)
np.mean(pred_RF_train==Y_train1)
pred_RF_test=classifier_RF.predict(X_test1)
np.mean(pred_RF_test==Y_test1)
#Train Accuracy RF=100
#Test Accuracy RF=86.60



positive=Reviews[Reviews["Sentiment"]=="positive"]
negative=Reviews[Reviews["Sentiment"]=="negative"]

print(positive.shape,negative.shape)

#Performing under sampling#
from imblearn.under_sampling import NearMiss

nm = NearMiss()
X1,Y1=nm.fit_resample(X,Y)

X_train, X_test, Y_train, Y_test = train_test_split(X1, Y1, test_size = 0.3)

classifier_NB = GaussianNB()
classifier_NB.fit(X_train, Y_train)
pred_NB_train=classifier_NB.predict(X_train)
np.mean(pred_NB_train==Y_train)
pred_NB_test=classifier_NB.predict(X_test)
np.mean(pred_NB_test==Y_test)
#Train Accuracy NB=94.75
#Test Accuracy NB=63.91

confusion_matrix(pred_NB_test,Y_test)
cm_classifier_NB = confusion_matrix(Y_test,pred_NB_test)
#print('Confusion matrix\n', cm)
cm_matrix_NB = pd.DataFrame(data=cm_classifier_NB, columns=['True Negative',  'True Positive'], 
                        index=['Predicted Negative', 'Predicted Positive'])
sns.heatmap(cm_matrix_NB, annot=True, fmt='d', cmap='YlGnBu')
plt.show()

classifier_MNB = MultinomialNB()
classifier_MNB.fit(X_train, Y_train)
pred_MNB_train=classifier_MNB.predict(X_train)
np.mean(pred_MNB_train==Y_train)
pred_MNB_test=classifier_MNB.predict(X_test)
np.mean(pred_MNB_test==Y_test)
#Train Accuracy MNB=90.30
#Test Accuracy MNB=85.74



classifier_DT = DecisionTreeClassifier(criterion = "entropy", random_state = 0)
classifier_DT.fit(X_train,Y_train)
pred_DT_train=classifier_DT.predict(X_train)
np.mean(pred_DT_train==Y_train)
pred_DT_test=classifier_DT.predict(X_test)
np.mean(pred_DT_test==Y_test)
#Train Accuracy DT=100
#Test Accuracy DT=75.01


classifier_LR=LogisticRegression()
classifier_LR.fit(X_train,Y_train)
pred_LR_train=classifier_LR.predict(X_train)
np.mean(pred_LR_train==Y_train)
pred_LR_test=classifier_LR.predict(X_test)
np.mean(pred_LR_test==Y_test)
#Train Accuracy LR=94.35
#Test Accuracy LR=90.04




classifier_ADA=AdaBoostClassifier()
classifier_ADA.fit(X_train,Y_train)
pred_ADA_train=classifier_ADA.predict(X_train)
np.mean(pred_ADA_train==Y_train)
pred_ADA_test=classifier_ADA.predict(X_test)
np.mean(pred_ADA_test==Y_test)
#Train Accuracy ADA=86.38
#Test Accuracy LR=84.78



classifier_RF=RandomForestClassifier()
classifier_RF.fit(X_train,Y_train)
pred_RF_train=classifier_RF.predict(X_train)
np.mean(pred_RF_train==Y_train)
pred_RF_test=classifier_RF.predict(X_test)
np.mean(pred_RF_test==Y_test)
#Train Accuracy RF=100
#Test Accuracy RF=86.41


classifier_SVM=SVC()
classifier_SVM.fit(X_train,Y_train)
pred_SVM_train=classifier_SVM.predict(X_train)
np.mean(pred_SVM_train==Y_train)
pred_SVM_test=classifier_SVM.predict(X_test)
np.mean(pred_SVM_test==Y_test)
#Train Accuracy SVM=99.06
#Test Accuracy SVM=91.55

classifier_KNN=KNeighborsClassifier()
classifier_KNN.fit(X_train,Y_train)
pred_KNN_train=classifier_KNN.predict(X_train)
np.mean(pred_KNN_train==Y_train)
pred_KNN_test=classifier_KNN.predict(X_test)
np.mean(pred_KNN_test==Y_test)
#Train Accuracy SVM=77.53
#Test Accuracy SVM=70.71

#Hyper parameter tuning of SVM
from sklearn.model_selection import GridSearchCV
parameters=[{'C':[1,10,100,1000],'kernel':['linear']},{'C':[1,10,100,1000],'kernel':['rbf'],'gamma':[0.1,0.2,0.3,0.4,0.5,0.6,0.8,0.9]}]
grid_search=GridSearchCV(estimator=classifier_SVM,param_grid=parameters,scoring='accuracy',cv=10,n_jobs=-1)

grid=grid_search.fit(X_train,Y_train)
grid.best_params_

classifier_SVM_tuned=SVC(kernel="linear",C=10)
classifier_SVM_tuned.fit(X_train,Y_train)
pred_SVM_train=classifier_SVM_tuned.predict(X_train)
np.mean(pred_SVM_train==Y_train)
pred_SVM_test=classifier_SVM_tuned.predict(X_test)
np.mean(pred_SVM_test==Y_test)
#Train Accuracy SVM=99.24
#Test Accuracy SVM=90.04


#Hyper parameter tuning of KNN
num_folds = 10
seed = 7
scoring = 'accuracy'

from sklearn.neighbors import KNeighborsClassifier
classifier_KNN=KNeighborsClassifier()
neighbors = [1,3,5,7,9,11,13,15,17,19,21]
param_grid = dict(n_neighbors=neighbors)

kfold = KFold(n_splits=num_folds, random_state=seed)
grid_model_knn = GridSearchCV(estimator=classifier_KNN, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_model_knn.best_params_


classifier_KNN_tuned=KNeighborsClassifier(n_neighbors=9)
classifier_KNN_tuned.fit(X_train,Y_train)
pred_KNN_train=classifier_KNN_tuned.predict(X_train)
np.mean(pred_KNN_train==Y_train)
pred_KNN_test=classifier_KNN_tuned.predict(X_test)
np.mean(pred_KNN_test==Y_test)
#Train Accuracy SVM=88.23
#Test Accuracy SVM=75.81


#Hyper parameter tuning of Logistic Regression
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import GridSearchCV 
  
# Creating the hyperparameter grid 
c_space = np.logspace(-5, 8, 15) 
param_grid = {'C': c_space} 
  
# Instantiating logistic regression classifier 
classifier_LR = LogisticRegression() 
  
# Instantiating the GridSearchCV object 
grid_search_KNN= GridSearchCV(classifier_LR, param_grid, cv = 5) 
  
grid_search_KNN.fit(X_train,Y_train) 
grid_search_KNN.best_params_


classifier_LR_tuned=LogisticRegression(C=2275.845926074791)
classifier_LR_tuned.fit(X_train,Y_train)
pred_LR_train=classifier_LR_tuned.predict(X_train)
np.mean(pred_LR_train==Y_train)
pred_LR_test=classifier_LR_tuned.predict(X_test)
np.mean(pred_LR_test==Y_test)
#Train Accuracy LR=100
#Test Accuracy LR=88.69s


#Model Deployement#




df.to_excel('data.xlsx')
