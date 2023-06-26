import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer

#nltk.download('punkt') -> Uncomment this if You didn't download it
stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentece, all_words):
    tokenized_sentece = [stem(w) for w in tokenized_sentece]
    
    bag = np.zeros(len(all_words), dtype=np.float32)
    for index, w in enumerate(all_words):
        if w in tokenized_sentece:
            bag[index] = 1.0
            
    return bag        

