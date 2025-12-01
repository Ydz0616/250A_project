import numpy as np
import pandas as pd
from typing import List, Dict
import random
from sklearn.ensemble import RandomForestClassifier

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

        if 7 <= hour < 10:
            return "work"
        if 10 <= hour < 12:
            return "home"
        if 12 <= hour < 13:
            return "eat"
        if 13 <= hour < 14:
            return "work"
        if 14 <= hour < 17:
            return "work"
        if 17 <= hour < 19:
            return "eat"
        if 19 <= hour < 22:
            return "leisure"

        return "home"

class FrequencyBaseline:
    """
    Baseline that predicts purpose based on the observed frequency of (mode -> purpose) in training.

    Data input is just the whole training data; then the model gets normliazed purpose distribution per mode; 

    Final prediction is random sampling w/ purpose distribution as weights
    """
    def __init__(self):
        self.mode_purpose_probs: Dict[str, Dict[str, float]] = {}
        self.most_common_purpose = "home"
        
    def fit(self, train_sequences: List[List[tuple]]):
        """
        Learn the distribution of purposes for each mode from the training data.
        
        Args:
            train_sequences: List of sequences, where each sequence is a list of (mode, purpose) tuples.
        """
        
        counts = {} # {mode:{purpose,count}}; for every mode; keeps track of purpose and count
        purpose_count = {} # {purpose:count} ; keeps track of purpose count 
        for seq in train_sequences:
            for mode,purpose in seq:
                purpose_count[purpose] = purpose_count.get(purpose,0) +1

                if mode not in counts:
                    counts[mode] = {}
                counts[mode][purpose] = counts[mode].get(purpose,0) +1

        # most commmon purpose
        if purpose_count:
            self.most_common_purpose = max(purpose_count,key=purpose_count.get)

        #  normalize
        for mode,purpose_count in counts.items():
            total_per_mode = sum(purpose_count.values())
            self.mode_purpose_probs[mode] = {}

            for purpose,count in purpose_count.items():
                self.mode_purpose_probs[mode][purpose] = float( count/total_per_mode )


    def predict(self, mode: str) -> str:
        """
        Randomly sample a purpose given the mode using the learned probabilities.
        """
        if mode not in self.mode_purpose_probs:
            return self.most_common_purpose
        
        prob_dict = self.mode_purpose_probs[mode]
        purposes = list(prob_dict.keys())
        probs = list(prob_dict.values())

        # weighted random sampling; take 1; given probs ( normalized ) as weights
        predicted_purpose = random.choices(purposes, weights=probs, k=1)[0]
        
        return predicted_purpose

class RFBaseline:
    """
    Baseline that uses a random forest classifier to predict the next purpose

    Data input is not sequential, instead, is (mode,purpose)

    """

    def __init__(self,n_estimators=300, random_seed = 42):
        self.clf = RandomForestClassifier(n_estimators=n_estimators,random_state=random_seed)
    
    def fit(self, X:np.ndarray, y:np.ndarray):
        self.clf.fit(X,y)

    def predict(self, X:np.ndarray):
        return self.clf.predict(X)





