from pathlib import Path
from typing import Dict, List, Tuple
from torch.utils.data import Dataset
import torch
import json

from components.config import DLTrainingConfig
from nltk_utils import tokenize, stem, bag_of_words

class ChatDataset(Dataset):
    def __init__(
        self,
        inputs: List[Path],
        cfg: DLTrainingConfig
        ) -> None:
        self.inputs = inputs
        self.cfg = cfg
        self.all_words, self.categories, self.xy: Tuple[List[str], List[str], List[str]] = self._initialize_all_words()
    
    def _initialize_all_words(self) -> Tuple[List[str], List[str], List[str]]:
        """Create and return lists of words.

        Returns:
            Tuple[List[str], List[str], List[str]]: Tuple containing list of words lists.
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
        
        return (all_words, categories, xy)
        
    def __len__(self) -> int:
        """Get the number of samples in dataset.

        Returns:
            int: Number of samples in dataset.
        """
        return len()
    # dataset[index]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    