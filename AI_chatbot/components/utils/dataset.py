from pathlib import Path
from typing import List, Tuple
from torch.utils.data import Dataset
import numpy as np
import torch
import json

from components.config import DLTrainingConfig
from components.utils.helpers import WordsCollection
from nltk_utils import tokenize, stem, bag_of_words

class ChatDataset(Dataset):
    """Custrom dataset class for handling input intents.json files.

    Args:
        inputs (List[Path]): List of file paths for input intents files.
        cfg (DLTrainingConfig): Training configuration object.
    """
    def __init__(
        self,
        inputs: List[Path],
        cfg: DLTrainingConfig
        ) -> None:
        self.inputs = inputs
        self.cfg = cfg
        self.words_collection: WordsCollection = self._set_words_collection()
    
    def _set_words_collection(self) -> WordsCollection:
        """Create and return the WordsCollection object for dataset class.
        This method constructs and organizes the words for the dataset.

        Returns:
            WordsCollection: An object containing dataset words related information.
        """
        categories = []
        all_words = []
        xy = []
        
        # Loop through all intents files and fill lists
        for path in self.inputs:
            with open(path, 'r') as f:
                intents = json.load(f)
                for intent in intents['intents']:
                    category = intent['category']
                    categories.append(category)
                    for question in intent['Question']:
                        w = tokenize(question)
                        all_words.extend(w)
                        xy.append((w, category))
        
        # Stem all words and sort lists                
        all_words = [stem(w) for w in all_words if w not in self.cfg.ignore_words]
        all_words = sorted(set(all_words))
        categories = sorted((set(categories)))
        
        words_collection = WordsCollection(
            categories=categories,
            all_words=all_words,
            xy=xy
        )
        
        return words_collection
        
    def __len__(self) -> int:
        """Get the number of samples in dataset.

        Returns:
            int: Number of samples in dataset.
        """
        return len(self.words_collection.all_words)
    
    @property
    def get_categories(self) -> int:
        """Get the number of categories in dataset.

        Returns:
            int: Number of categories in dataset.
        """
        return len(self.words_collection.categories)
    
    def __getitem__(self, index: int) -> Tuple[np.array, np.array]:
        """Get a single sample from the dataset.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            Tuple[np.array, np.array]: A tuple containing the input data and corresponding category.
        """
        X_data = []
        y_data = []
        for (pattern_sentence, tag) in self.words_collection.xy:
            bag = bag_of_words(pattern_sentence, self.words_collection.all_words)
            X_data.append(bag)    
            
            label = self.words_collection.categories.index(tag)
            y_data.append(label)
        
        X_data = np.array(X_data)
        y_data = np.array(y_data)
        
        return (X_data[index], y_data[index])