import numpy as np
import pandas as pd
from typing import List, Dict
import random

class TimeOfDayBaseline:
    """
    Rule-based baseline that predicts purpose based on the time of day.
    
    Rules:
    - 7-10 AM: work
    - 10-12 AM: home
    - 12-1 PM: eat
    - 1-2 PM: work
    - 2-5 PM: work
    - 5-7 PM: eat
    - 7-10 PM: leisure
    """
    def __init__(self):
        pass
        
    def predict(self, timestamp: pd.Timestamp) -> str:
        """
        Predict purpose based on the hour of the timestamp.
        """
        hour = timestamp.hour
        
        # TODO: Implement the logic
        # if 7 <= hour < 10: return "work"
        # ...
        
        return "home" # Default fallback

class FrequencyBaseline:
    """
    Baseline that predicts purpose based on the observed frequency of (mode -> purpose) in training.
    """
    def __init__(self):
        self.mode_purpose_probs: Dict[str, Dict[str, float]] = {}
        
    def fit(self, train_sequences: List[List[tuple]]):
        """
        Learn the distribution of purposes for each mode from the training data.
        
        Args:
            train_sequences: List of sequences, where each sequence is a list of (mode, purpose) tuples.
        """
        # TODO: Count occurrences of each purpose given a mode
        
        # TODO: Normalize counts to get probabilities P(purpose | mode)
        
        pass
        
    def predict(self, mode: str) -> str:
        """
        Randomly sample a purpose given the mode using the learned probabilities.
        """
        if mode not in self.mode_purpose_probs:
            # TODO: Handle unseen modes (e.g., return most common purpose overall)
            return "home"
            
        # TODO: Sample from self.mode_purpose_probs[mode]
        
        return "home"

