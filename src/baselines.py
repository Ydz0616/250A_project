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
    """
    def __init__(self):
        self.mode_purpose_probs = {} #mode -> {purpose -> probability}
        self.global_counts = {} #purpose -> count (for unseen modes)
        
    def fit(self, train_sequences: List[List[tuple]]):
        """
        Learn the distribution of purposes for each mode from the training data.
        
        Args:
            train_sequences: List of sequences, where each sequence is a list of (mode, purpose) tuples.
        """
        counts = {} #mode -> {purpose -> count}
        global_counts = {} #purpose -> total count
        for seq in train_sequences:
            for mode, purpose in seq:
                if mode not in counts:
                    counts[mode] = {}
                if purpose not in counts[mode]:
                    counts[mode][purpose] = 0
                counts[mode][purpose] += 1

                if purpose not in global_counts:
                    global_counts[purpose] = 0
                global_counts[purpose] += 1
        
        mode_purpose_probs = {}
        for mode in counts:
            mode_purpose_probs[mode] = {}
            total = sum(counts[mode].values())
            for purpose in counts[mode]:
                mode_purpose_probs[mode][purpose] = counts[mode][purpose] / total
        
        self.mode_purpose_probs = mode_purpose_probs
        self.global_counts = global_counts
        
    def predict(self, mode: str) -> str:
        """
        Randomly sample a purpose given the mode using the learned probabilities.
        """
        probs = self.mode_purpose_probs[mode]
        r = random.random()
        cumulative = 0
        
        #Build a number line
        for purpose, p in probs.items():
            cumulative += p
            if r <= cumulative:
                return purpose
        
        return 'Home'

#Baseline tests - GPT generated
if __name__ == "__main__":
    print("=== Testing TimeOfDayBaseline ===")
    tod = TimeOfDayBaseline()

    timestamps = [
        pd.Timestamp("2024-01-01 08:00"),  # work
        pd.Timestamp("2024-01-01 11:00"),  # home
        pd.Timestamp("2024-01-01 12:30"),  # eat
        pd.Timestamp("2024-01-01 14:30"),  # work
        pd.Timestamp("2024-01-01 18:00"),  # eat
        pd.Timestamp("2024-01-01 20:00"),  # leisure
        pd.Timestamp("2024-01-01 03:00")   # home (default)
    ]

    for t in timestamps:
        print(t, "→", tod.predict(t))


    print("\n=== Testing FrequencyBaseline ===")
    fb = FrequencyBaseline()

    # Simple training set
    train_sequences = [
        [("car", "work"), ("car", "work"), ("car", "home")],
        [("walk", "eat"), ("walk", "leisure")],
        [("bus", "home"), ("bus", "home"), ("bus", "errand")]
    ]

    fb.fit(train_sequences)

    print("Learned probabilities:")
    for mode, probs in fb.mode_purpose_probs.items():
        print(f"  {mode}: {probs}")

    # Fix randomness
    random.seed(0)

    print("\nPredictions:")
    print("car →", fb.predict("car"))
    print("walk →", fb.predict("walk"))
    print("bus →", fb.predict("bus"))
    print("train (unseen mode) →", fb.predict("train"))