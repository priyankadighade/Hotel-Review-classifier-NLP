# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 16:46:22 2020

@author: Irfan Sheikh
"""

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import os
os.chdir("D:\\Data Science\\Project\\Project Hotel Review Excelr")


df=pd.read_excel("D:\\Data Science\\Project\\Project Hotel Review Excelr\\hotel_reviews.xlsx")
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
from nltk.stem import WordNetLemmatizer 

lemmatizer = WordNetLemmatizer()
from nltk.tokenize import word_tokenize
nltk.download('punkt')
ps=PorterStemmer()
import nltk
nltk.download('wordnet')

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
STOP_WORDS |= {"nt","hotel","room"}
print(STOP_WORDS)

def text_clean(text):
    text=text.lower()
   
    text=re.sub("\[.*?\]","",text)
    text=re.sub("[%s]" % re.escape(string.punctuation),"",text)
    text=re.sub("\w*\d\w*","",text)
    text=re.sub("\n","",text)
    
    return text

cleaned1 = lambda x: text_clean(x)
df["Cleaned_Reviews"]=pd.DataFrame(df.Review.apply(cleaned1))


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





#Creating positive wordcloud
from wordcloud import WordCloud
  
with open("D:\\Data Science\\Assignment files\\Text Mining Assignment\\positive-words (1).txt","r") as pos:
    positive = pos.read().split("\n")
   
wordcloud1 = WordCloud(width = 3000, height = 2000, random_state=1, background_color='white', colormap='Set2', collocations=False, stopwords=positive).generate(str(corpus ))
plt.imshow(wordcloud1)


#Creating negative wordcloud
with open("D:\\Data Science\\Assignment files\\Text Mining Assignment\\negative-words.txt","r") as neg:
    negative = neg.read().split("\n")


wordcloud_neg = WordCloud(width = 3000, height = 2000, random_state=1, background_color='black', colormap='Pastel1', collocations=False,stopwords=negative).generate(str(corpus))
plt.imshow(wordcloud_neg)


#Polarity and subjectivity#
import textblob
from textblob import TextBlob

df["Polarity"]=df["Cleaned_Reviews"].apply(lambda x:TextBlob(x).sentiment.polarity)
df["Subjectivity"]=df["Cleaned_Reviews"].apply(lambda x:TextBlob(x).sentiment.subjectivity)


#Printing 5 reviews with highest polarity
print("5 Random Reviews with Highest Polarity:")
for index,review in enumerate(df.iloc[df['Polarity'].sort_values(ascending=False)[:5].index]['Cleaned_Reviews']):
    print('Review {}:\n'.format(index+1),review)
    
  
#Printing 5 reviews with negative polarity  
print("5 Random Reviews with Lowest Polarity:")
for index,review in enumerate(df.iloc[df['Polarity'].sort_values(ascending=True)[:5].index]['Cleaned_Reviews']):
  print('Review {}:\n'.format(index+1),review)    
    
 #Frequency Distribution based on Polarity
plt.figure(figsize=(30,20),facecolor="green",edgecolor="orange")
plt.margins(0.02)
plt.xlabel("Polarity",fontsize=50)
plt.xticks(fontsize=50)
plt.ylabel("Frequency",fontsize=50)
plt.yticks(fontsize=50)
plt.hist(df["Polarity"],bins=50)
plt.title("Frequency Distribution based on Polarity",fontsize=60)
plt.show()


#Pie plot of percentage of ratings
plt.figure(figsize=(30,10))
plt.title('Percentage of Ratings', fontsize=20)
df.Rating.value_counts().plot(kind='pie', labels=['Rating5', 'Rating4', 'Rating3', 'Rating2', 'Rating1'],
                              wedgeprops=dict(width=.7), autopct='%1.0f%%', startangle= -20, 
                              textprops={'fontsize': 15})



#Sentiment distribution based on ratings
polarity_avg = df.groupby('Rating')['Polarity'].mean().plot(kind='bar', figsize=(50,30),color="green",edgecolor="orange")
plt.xlabel('Rating', fontsize=45)
plt.ylabel('Average Sentiment', fontsize=45)
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)
plt.title('Average Sentiment per Rating Distribution', fontsize=50)
plt.show()

#Counting no of words in each review
df["Word Count"]=df["Cleaned_Review_Lemmatized"].apply(lambda x:len(str(x).split()))

#Average no of words wrt ratings
word_avg=df.groupby("Rating")["Word Count"].mean().plot(kind="bar",figsize=(50,30),color="green",edgecolor="orange")
plt.xlabel('Rating',fontsize=35)
plt.ylabel("Average Count of Words",fontsize=35)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.title("Average Count of Words wrt Ratings",fontsize=50)
plt.show()

#Counting no of letters in each review
df['review_len'] = df['Cleaned_Review_Lemmatized'].astype(str).apply(len)


#Average no of letters wrt ratings
letter_avg = df.groupby('Rating')['review_len'].mean().plot(kind='bar', figsize=(50,30),color="green",edgecolor="orange")
plt.xlabel('Rating', fontsize=35)
plt.ylabel('Count of Letters in Rating', fontsize=35)
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)
plt.title('Average Number of Letters wrt Rating ', fontsize=40)
plt.show()


#Corelation of features
corelation=df[["Rating","Polarity","Word Count","review_len","Subjectivity"]].corr()

sns.heatmap(corelation,xticklabels=corelation.columns,yticklabels=corelation.columns,annot=True)


#Counting 100 most common words in our data
from nltk.probability import FreqDist
mostcommon = FreqDist(df["Cleaned_Review_Lemmatized"]).most_common(100)
wordcloud = WordCloud(width=1600, height=800, background_color='white').generate(str(corpus))
fig = plt.figure(figsize=(30,10), facecolor='white')
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title('Top 100 Most Common Words', fontsize=100)
plt.tight_layout(pad=0)
plt.show()



polarity_positive_data=pd.DataFrame(df.groupby("Cleaned_Reviews")["Polarity"].mean().sort_values(ascending=True))

plt.figure(figsize=(50,30))
plt.xlabel('Polarity',fontsize=30)
plt.ylabel('Reviews',fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.title('Polarity of Reviews',fontsize=50)
polarity_graph=plt.barh(np.arange(len(polarity_positive_data.index)),polarity_positive_data['Polarity'],color='purple',)


#categorizing reviews in positive,negative and neutral#
def sentiment(x):
    if x<0:
        return 'negative'
    elif x==0:
        return 'neutral'
    else:
        return 'positive'
    
df['polarity_score']=df['Polarity'].\
   map(lambda x: sentiment(x))

plt.hist(df["polarity_score"],bins=10, color='green', alpha=0.8, label='Value', edgecolor='orange', linewidth=2)
plt.xlabel("Polarity Sentiments",fontsize=35)
plt.ylabel("Frequency",fontsize=35)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.title("Polarity Sentiments wrt Frequency",fontsize=60)
plt.show()




from collections import Counter


#Most frequent words for rating 1
group_by = df.groupby('Rating')['Cleaned_Review_Lemmatized'].apply(lambda x: Counter(' '.join(x).split()).most_common(25))
group_by_0 = group_by.iloc[0]
words0 = list(zip(*group_by_0))[0]
freq0 = list(zip(*group_by_0))[1]
plt.figure(figsize=(50,30))
plt.bar(words0, freq0,color="green",edgecolor="orange")
plt.xlabel('Words', fontsize=50)
plt.ylabel('Frequency of Words', fontsize=50)
plt.yticks(fontsize=40)
plt.xticks(rotation=60, fontsize=40)
plt.title('Frequency of 25 Most Common Words for Rating=1', fontsize=60)
plt.show()

#Most frequent words for rating 2
group_by_1 = group_by.iloc[1]
words1 = list(zip(*group_by_1))[0]
freq1 = list(zip(*group_by_1))[1]
plt.figure(figsize=(50,30))
plt.bar(words1, freq1,color="green",edgecolor="orange")
plt.xlabel('Words', fontsize=50)
plt.ylabel('Frequency of Words', fontsize=50)
plt.yticks(fontsize=40)
plt.xticks(rotation=60, fontsize=40)
plt.title('Frequency of 25 Most Common Words for Rating=2', fontsize=60)
plt.show()

#Most frequent words for rating 3
group_by_2 = group_by.iloc[2]
words2 = list(zip(*group_by_2))[0]
freq2 = list(zip(*group_by_2))[1]
plt.figure(figsize=(50,30))
plt.bar(words2, freq2,color="green",edgecolor="orange")
plt.xlabel('Words', fontsize=50)
plt.ylabel('Frequency of Words', fontsize=50)
plt.yticks(fontsize=40)
plt.xticks(rotation=60, fontsize=40)
plt.title('Frequency of 25 Most Common Words for Rating=3', fontsize=60)
plt.show()


#Most frequent words for rating 4
group_by_3 = group_by.iloc[3]
words3 = list(zip(*group_by_3))[0]
freq3 = list(zip(*group_by_3))[1]
plt.figure(figsize=(50,30))
plt.bar(words3, freq3,color="green",edgecolor="orange")
plt.xlabel('Words', fontsize=50)
plt.ylabel('Frequency of Words', fontsize=50)
plt.yticks(fontsize=40)
plt.xticks(rotation=60, fontsize=40)
plt.title('Frequency of 25 Most Common Words for Rating=4', fontsize=60)
plt.show()


#Most frequent words for rating 5
group_by_4 = group_by.iloc[4]
words4 = list(zip(*group_by_4))[0]
freq4 = list(zip(*group_by_4))[1]
plt.figure(figsize=(50,30))
plt.bar(words4, freq4,color="green",edgecolor="orange")
plt.xlabel('Words', fontsize=50)
plt.ylabel('Frequency of Words', fontsize=50)
plt.yticks(fontsize=40)
plt.xticks(rotation=60, fontsize=40)
plt.title('Frequency of 25 Most Common Words for Rating=5', fontsize=60)
plt.show()














#topic modelling using LDA#

import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
%matplotlib inline

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(df["Cleaned_Review_Lemmatized"]))

print(data_words)

# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)
print(trigram_mod)
# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
    

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]
    

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]
    

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out
    
    


data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('en', disable=['parser', 'ner'])

import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()
# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(data_lemmatized[:1])



# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
print(corpus[:1])

# Human readable format of corpus (term-frequency)
[[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]


lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=10, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]


import pyLDAvis.gensim
import pickle 
import pyLDAvis
# Visualize the topics
pyLDAvis.enable_notebook()
LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
LDAvis_prepared