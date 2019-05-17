# Phase II
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
from nltk import ngrams
from nltk.probability import FreqDist
from nltk.collocations import BigramCollocationFinder
import re

reviews = pd.read_csv("yelp.csv")

stop_words =set(stopwords.words("english"))
ps = PorterStemmer()

def preprocess(text):
    text = re.sub(r'[^\w\s]','',text) # removing non alphanumeric characters and spaces
    tokenized = nltk.word_tokenize(text) # tokenizing
    filtered_sentence = [] # empty array of filtered sentence
    for word in tokenized: # iterate through all the words in tokenized array
        if word not in stop_words: # if it is not a stop word
            filtered_sentence.append(word.lower()) # put it into the filtered_sentence array
            
    return filtered_sentence

preprocessed_text = []

#for text in reviews:
#    preprocessed_text = preprocess(text)

for text in reviews['text']:
    preprocessed_text.append(preprocess(text))

print("\n")
#print(preprocessed_text)
print("\n")

def most_frequent(tokenized_text,n):
    stemmed_list = list()
    for word in tokenized_text:
        stemmed_list.append(word.lower())
    return FreqDist(stemmed_list).most_common(n)

print("\n")
for text in preprocessed_text:
    print(most_frequent(text,5))
print("\n")

def displayNgrams(tokenized_text, n):
    return ngrams(tokenized_text,n)

print("\n")
for text in preprocessed_text:
    for x in displayNgrams(text,2):
        print(x)
  
print("\n")

def mostFreqBigram(freq_bi_gram,number_of_bigram,text):
    bi_gram_freq_list = FreqDist(displayNgrams(text,2)).most_common(len(list(displayNgrams(text,2))))
    result = list()
    i=0
    for x in bi_gram_freq_list:
        if x[1] == freq_bi_gram and i<number_of_bigram:
            result.append(x[0])
            i+=1
            
    return result

print("\n")
for text in preprocessed_text:
    print(mostFreqBigram(2,3,text))
print("\n")

def probable_occur(bi_gram):
    bi_gram_measures = nltk.collocations.BigramAssocMeasures()
    finder = BigramCollocationFinder.from_documents(bi_gram)
    return sorted(finder.nbest(bi_gram_measures.pmi,10))
    
    
print("\n")
for text in preprocessed_text:
    print(probable_occur(displayNgrams(text,2)))
print("\n")
    
def score_bi_gram(bi_gram):
    bi_gram_measures = nltk.collocations.BigramAssocMeasures()
    finder = BigramCollocationFinder.from_documents(bi_gram)
    finder.apply_freq_filter(2)
    scored = finder.score_ngrams(bi_gram_measures.pmi)
    return scored

print("\n")
for text in preprocessed_text:
    print(score_bi_gram(displayNgrams(text,2)))
print("\n")
    
def pos_tag(text):
    tokenize = nltk.word_tokenize(text)
    return nltk.pos_tag(tokenize)
    
    
print("\n")
for text in reviews['text']:
    print(pos_tag(text))
print("\n")
    
    
def numOfTags(tagged_text):
    words = {}
    for x in tagged_text:
        if x[1] in words:
            words[x[1]] += 1
        else:
            words[x[1]] = 1
    return sorted(words.items(), key=lambda x: x[1], reverse=True)[:10]
    
print("\n")
for text in reviews['text']:
    print(numOfTags(pos_tag(text)))
print("\n")

def get_specific_tag(tagged_text , tag):
    result = list()
    for k,v in tagged_text:
        if tag in v:
            result.append(k)
    return result
    

print("\n")
for text in reviews['text']:
    print(get_specific_tag(pos_tag(text),"NNP"))
print("\n")