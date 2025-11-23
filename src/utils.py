import numpy as np
from typing import List, Dict, Iterable

class LabelEncoder:
    """
    Helper to encode string labels to integers and decode them back.
    """
    def __init__(self):
        self.label_to_idx: Dict[str, int] = {}
        self.idx_to_label: Dict[int, str] = {}
        
    def fit(self, labels: Iterable[str]):
        """
        Learn the mapping from unique labels.
        """
        unique_labels = sorted(list(set(labels)))
        self.label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        self.idx_to_label = {i: label for i, label in enumerate(unique_labels)}
        
    def transform(self, labels: List[str]) -> np.ndarray:
        """
        Convert list of strings to numpy array of integers.
        """
        return np.array([self.label_to_idx[l] for l in labels])
    
    def inverse_transform(self, indices: List[int]) -> List[str]:
        """
        Convert integers back to strings.
        """
        return [self.idx_to_label[i] for i in indices]
        
    @property
    def classes_(self) -> List[str]:
        return [self.idx_to_label[i] for i in range(len(self.label_to_idx))]

    def __len__(self):
        return len(self.label_to_idx)

def calculate_accuracy(true_sequences: List[List[str]], predicted_sequences: List[List[str]]) -> float:
    """
    Calculate the element-wise accuracy of predicted sequences against ground truth.
    
    Args:
        true_sequences: List of lists containing ground truth labels.
        predicted_sequences: List of lists containing predicted labels.
        
    Returns:
        float: Accuracy (0.0 to 1.0).
    """
    # TODO: Flatten both lists
    
    # TODO: Compare element-wise
    
    # TODO: Compute ratio of matches to total elements
    
    pass

