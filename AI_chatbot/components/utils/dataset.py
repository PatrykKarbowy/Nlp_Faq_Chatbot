from pathlib import Path
from typing import List
from torch.utils.data import Dataset
import numpy as np
import json

from components.config import DLTrainingConfig
from .helpers import WordsCollection, LabeledCollection
from .nltk_utils import tokenize, stem, bag_of_words

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
        self.labeled_collection: LabeledCollection = self._set_labeled_collection()
    
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
    
    def _set_labeled_collection(self) -> LabeledCollection:
        """Create and return the LabeledCollection object for dataset class.
        This method constructs and organizes the splitted data for the dataset.

        Returns:
            WordsCollection: An object containing X and y data.
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
        
        labeled_collection = LabeledCollection(
            X_data=X_data,
            y_data=y_data
        )
        
        return labeled_collection
        
    def __len__(self) -> int:
        """Get the number of samples in dataset.

        Returns:
            int: Number of samples in dataset.
        """
        return len(self.labeled_collection.X_data)
    
    @property
    def get_categories(self) -> int:
        """Get the number of categories in dataset.

        Returns:
            int: Number of categories in dataset.
        """
        return len(self.words_collection.categories)
    
    @property
    def get_inputs_number(self) -> int:
        """Get the number of inputs for the model.

        Returns:
            int: Number of inputs for the model.
        """
        return len(self.words_collection.all_words)   
    
    def __getitem__(self, index: int) -> tuple[np.array, np.array]:
        """Get a single sample from the dataset.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple[np.array, np.array]: A tuple containing the input data and corresponding category.
        """
        return (self.labeled_collection.X_data[index], self.labeled_collection.y_data[index])