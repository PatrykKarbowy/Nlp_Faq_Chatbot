from typing import List

import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer

#nltk.download('punkt') -> Uncomment this if You didn't download it
stemmer = PorterStemmer()

def tokenize(sentence: str) -> List[str]:
    """Function that create and return list of tokenized words from sentece.

    Args:
        sentence (str): A sentence to be tokenized.

    Returns:
        List[str]: Tokenized sentence.
    """
    return nltk.word_tokenize(sentence)

def stem(word: str) -> str:
    """Function that create and return stemmed word.

    Args:
        word (str): A word to be stemmed.

    Returns:
        str: Stemmed word.
    """
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentece: List[str], all_words: List[str]) -> np.array:
    """Function that create and return bag of words numpy list.

    Args:
        tokenized_sentece (List[str]): Tokenized sentence.
        all_words (List[str]): List of all words.

    Returns:
        np.array: Unordered collection of words (Bag of words).
    """
    tokenized_sentece = [stem(w) for w in tokenized_sentece]
    
    bag = np.zeros(len(all_words), dtype=np.float32)
    for index, w in enumerate(all_words):
        if w in tokenized_sentece:
            bag[index] = 1.0
            
    return bag        

