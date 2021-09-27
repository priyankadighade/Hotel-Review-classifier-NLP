# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 11:05:21 2020

@author: Irfan Sheikh
"""
import pandas as pd
import numpy as np
from flask import Flask,render_template,url_for,request
from nltk.stem.porter import PorterStemmer
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pickle






def remove_pattern(input_txt,pattern):
    r= re.findall(pattern,input_txt)
    
    for i in r:
        input_txt=re.sub(i,"",input_txt)
        return input_txt
    
def count_punct(text):
    count=sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text)-text.count("")),3)*100


    

data=pd.read_excel("data.xlsx")
data.Columns=["Sentiment","Review"]
#features and lables
data["Rating_Senti"]=data["Sentiment"].map({"positive":3,"negative":1,"neutral":2})
data["tidy_Reviews_"]=np.vectorize(remove_pattern)(data["Review"],"@[\w]*")
tokenized_Reviews= data["Cleaned_Reviews"].apply(lambda x:x.split())
stemmer=PorterStemmer()
tokenized_Reviews= tokenized_Reviews.apply(lambda x: [stemmer.stem(i) for i in x])

for i in range(len(tokenized_Reviews)):
    tokenized_Reviews[i]="".join(tokenized_Reviews[i])
    
data["tidy_Reviews"]=tokenized_Reviews
data["body_len"]=data["tidy_Reviews"].apply(lambda x:len(x)-x.count(""))
data["punct%"]=data["tidy_Reviews"].apply(lambda x:count_punct(x))
cv=CountVectorizer(max_features=35000)
X=data["Cleaned_Reviews"]
X=cv.fit_transform(X)
Y=data["Rating_Senti"]

X=pd.concat([data["body_len"],data["punct%"],pd.DataFrame(X.toarray())],axis=1)

clf=LogisticRegression()
clf.fit(X,Y)


app=Flask(__name__)




@app.route("/")
def home():
    return render_template("Sample.html")

@app.route('/predict',methods=["POST"])
def predict():
    if request.method == "POST":
        message=request.form["message"]
        data= [message]
        vect= pd.DataFrame(cv.transform(data).toarray())
        body_len= pd.DataFrame([len(data) - data.count(" ")])
        punct = pd.DataFrame([count_punct(data)])
        total_data= pd.concat([body_len,punct,vect],axis = 1)
        my_prediction= clf.predict(total_data)
        
    return render_template("Result.html",prediction=my_prediction)        

if __name__ == '__main__':
    app.run(debug=True,use_reloader=False)



