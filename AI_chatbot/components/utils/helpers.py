from dataclasses import dataclass
from typing import Optional, List, Tuple

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