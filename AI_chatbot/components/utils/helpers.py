from dataclasses import dataclass
from typing import Optional, List, Tuple
from torch.utils.data import DataLoader

@dataclass
class WordsCollection:
    """A data class that holds varius lists used in dataset.
    
    Args:
        categories (Optional[List[str]]): Training data categories.
        all_words: (Optional[List[str]]): All words from training data.
        xy: (Optional[List[Tuple[str, str]]]): Categories and All words tuple.
    """
    
    categories: Optional[List[str]]
    all_words: Optional[List[str]]
    xy: Optional[List[Tuple[str, str]]]
    
@dataclass
class DataCollection:
    """A data class that holds varius data loaders and related information.
    
    Args:
        train_loader (Optional[DataLoader]): DataLoader for training data.
        val_loader: (Optional[DataLoader]): DataLoader for validation data.
        test_loader: (Optional[DataLoader]): DataLoader for test data.
    """
    
    train_loader: Optional[DataLoader]
    val_loader: Optional[DataLoader]
    test_loader: Optional[DataLoader]