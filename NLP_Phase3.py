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


def preprocess(text):
    text = re.sub(r'[^\w\s]','',text) # removing non alphanumeric characters and spaces
    tokenized = nltk.word_tokenize(text) # tokenizing
    filtered_sentence = [] # empty array of filtered sentence
    for word in tokenized:    
        filtered_sentence.append(word.lower()) # put it into the filtered_sentence array
    return filtered_sentence

preprocessed_text = []


#for text in reviews['text']:
for word in preprocess(reviews['text'][0]):
    preprocessed_text.append(word)

# Pos tagging
def pos_tag(text):
    tokenize = nltk.word_tokenize(text)
    return nltk.pos_tag(tokenize)

tagged_word_list = []
for word in preprocessed_text:   
    tagged_word_list.append(pos_tag(word))
    

# 1st Question
lexicon = {}

def construct_lexicon(tagged_word_list):
    for pair in tagged_word_list:
        key = pair[0][0]
        value = pair[0][1]
        if value in lexicon:
            if key not in lexicon[value]:
                lexicon[value].append(key)
        else:
            lexicon[value] = [key]
    return lexicon
        
     
print(construct_lexicon(tagged_word_list))
        
# 2nd Question

grammar = nltk.CFG.fromstring("""
S -> NP VP
PP -> P NP
NP -> Det N | Det N PP | 'I'
VP -> V NP | VP PP
Det -> 'an' | 'my'
N -> 'elephant' | 'pajamas'
V -> 'shot'
P -> 'in'
""")

def constructGrammar(sentence):
    parser = nltk.ChartParser(grammar)
    for tree in parser.parse(sentence):
      print(tree)
        
sampleSentence = 'I', 'shot', 'an', 'elephant', 'in', 'my', 'pajamas'
constructGrammar(sampleSentence)
