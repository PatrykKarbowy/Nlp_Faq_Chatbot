from dataclasses import dataclass
from typing import Optional, List, Tuple, Callable
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np

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
    """A data class that holds varius data loaders and related information from dataset.
    
    Args:
        train_loader (Optional[DataLoader]): DataLoader for training data.
        val_loader: (Optional[DataLoader]): DataLoader for validation data.
        test_loader: (Optional[DataLoader]): DataLoader for test data.
        num_categories: (int): Number of categories from dataset.
        num_inputs: (int): Number of inputs from dataset.
    """
    
    num_categories: int
    num_inputs: int
    train_loader: Optional[DataLoader]
    val_loader: Optional[DataLoader]
    test_loader: Optional[DataLoader] = None
    
@dataclass
class LabeledCollection:
    """A data class that holds X and y data.
    
    Args:
        X_data: (Optional[np.array]): X data used in dataset.
        y_data: (Optional[np.array]): Corresponding y data used in dataset.
    """
    
    X_data: Optional[np.array]
    y_data: Optional[np.array]
    
def build_loss_fn(name: str) -> Callable:
    """Build a loss function based on the provided name.

    Args:
        name (str): Name of the loss function to build.

    Returns:
        Callable: Loss function.
    """

    if name == "crossentropy":
        loss_fn = nn.CrossEntropyLoss()

    return loss_fn