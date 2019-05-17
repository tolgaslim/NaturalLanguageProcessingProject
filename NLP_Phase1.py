import nltk
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.probability import FreqDist
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

reviews = pd.read_csv("yelp.csv")
ps = PorterStemmer()
stop_words =set(stopwords.words("english"))



for text in reviews["text"]:
    stemmed_sentence = []
    tokenized = nltk.word_tokenize(text)
    for word in tokenized:
        if word not in stop_words:
            stemmed_sentence.append(ps.stem(word))



    distribution = FreqDist(stemmed_sentence)
    print(distribution.most_common(10))

    for a in tokenized:
        if len(a)>= 10:
            print(a)


    distribution.plot()