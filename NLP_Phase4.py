#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Created on Mon May 13 22:30:58 2019
    
    @author: arda tolga
    """

#importing libraries
import json
import re
import pandas as pd
import nltk
from nltk.probability import FreqDist
from nltk.corpus import stopwords
nltk.download("stopwords")
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import classification_report, confusion_matrix

nltk.download('punkt')
nltk.download('stopwords')

stop_words = stopwords.words("ENGLISH")


#reading dataset from csv file

df = pd.DataFrame.from_csv("yelp.csv")

#taking the reviews column
reviews = df.iloc[:,4]
points = df.iloc[:,2]


review_Array = []
for i in reviews:
    review_Array.append(i)

def method(index):
    review = re.sub('[^a-zA-Z]',' ',index)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    return review

corpus = []
for i in review_Array:
    corpus.append(method(i))
    

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=10000)
X = cv.fit_transform(corpus).toarray()
y = points

from sklearn.cross_validation import train_test_split
X_train, X_test , y_train, y_test =train_test_split(X,y,test_size = 0.30,random_state = 0)


print ("GaussianNB")

from sklearn.naive_bayes import GaussianNB
classifier= GaussianNB()
classifier.fit(X_train,y_train)
y_pred_NB = classifier.predict(X_test)


print("Accuracy score for NaviveByaes")

from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred_NB,y_test))
print(classification_report(y_test,y_pred_NB)) 



print("Linear Regression")

from sklearn.linear_model import LinearRegression 

clf = LinearRegression(normalize=True)
clf.fit(X_train,y_train)
y_pred_linear = clf.predict(X_test)


print("r^2 score for Linear")
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred_linear))

print("SVM")
from sklearn.svm import SVC
model = SVC(kernel = 'linear')
model.fit(X_train, y_train)
y_pred_SVM = model.predict(X_test)


print("confusion matrix for svm")

print(confusion_matrix(y_test,y_pred_SVM))  
print(accuracy_score(y_pred_SVM,y_test))
print(classification_report(y_test,y_pred_SVM)) 


print("KNN")
# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_knn = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_knn)

print(accuracy_score(y_pred_knn,y_test))





#******************************************************************************

#wordnet

