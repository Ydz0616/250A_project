import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any

# ==========================================
# Mappings
# ==========================================

MODE_MAPPING = {
    # Car
    "Mode::Car": "car",

    # Walk
    "Mode::Walk": "walk",

    # Bike
    "Mode::Bicycle": "bike",
    "Mode::Ebicycle": "bike",
    "Mode::MotorbikeScooter": "bike",

    # Bus
    "Mode::Bus": "bus",

    # Train (rail-based)
    "Mode::Train": "train",
    "Mode::RegionalTrain": "train",
    "Mode::LightRail": "train",
    "Mode::Tram": "train",
}

PURPOSE_MAPPING = {
    "Home": "home",
    "Work": "work",
    "Eat": "eat",

    # leisure group
    "Leisure": "leisure",
    "Sport": "leisure",
    "Family_friends": "leisure",
    "Shopping": "leisure",

    # errand group
    "Errand": "errand",
    "Assistance": "errand",
    "Wait": "errand",
}

def load_and_process_data(filepath: str) -> pd.DataFrame:
    """
    Load the dataset, apply mappings, and sort.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Processed dataframe with mapped 'mode' and 'purpose' columns,
                      sorted by user_id, event_date, and seq_idx.
    """
    # TODO: Load CSV
    # df = pd.read_csv(filepath)
    
    # TODO: Apply MODE_MAPPING to 'trip_mode' column to create 'mode' column
    
    # TODO: Apply PURPOSE_MAPPING to 'end_purpose' column to create 'purpose' column
    
    # TODO: Convert 'event_date' to datetime if needed, or ensure correct sorting
    
    # TODO: Sort by ['user_id', 'event_date', 'seq_idx']
    
    # TODO: Filter out any rows where mode or purpose is NaN after mapping (if any)
    
    pass

def create_user_sequences(df: pd.DataFrame) -> Dict[str, List[List[Tuple[str, str]]]]:
    """
    Group data into sequences of (Mode, Purpose) tuples per user.
    
    Args:
        df (pd.DataFrame): Sorted processed dataframe.

    Returns:
        Dict[str, List[List[Tuple[str, str]]]]: A dictionary where keys are user_ids
        and values are lists of sequences.
        Each sequence is a list of (mode, purpose) tuples corresponding to a day (or continuous block).
        
        Example structure:
        {
            'user_1': [
                [('car', 'work'), ('walk', 'eat'), ('car', 'home')],  # Day 1
                [('bus', 'leisure'), ('bus', 'home')]                # Day 2
            ],
            ...
        }
    """
    # TODO: Group by user_id
    
    # TODO: Inside each user, group by event_date to form daily sequences
    
    # TODO: Construct the list of (mode, purpose) tuples for each day
    
    pass

def train_test_split_by_user(sequences: Dict[str, Any], test_size: float = 0.2, random_seed: int = 42) -> Tuple[List[Any], List[Any]]:
    """
    Split data ensuring all sequences from a specific user go into the same split.
    
    Args:
        sequences (Dict): Dictionary of user sequences from create_user_sequences.
        test_size (float): Proportion of users to include in the test split.
        random_seed (int): Seed for reproducibility.

    Returns:
        Tuple[List, List]: (train_sequences, test_sequences)
        Each element is a flattened list of sequences (lists of tuples) suitable for HMM training/testing.
        We flatten the user structure because the HMM doesn't care about user ID, just the observed sequences.
    """
    # TODO: Get list of all unique user_ids
    
    # TODO: Shuffle user_ids
    
    # TODO: Split user_ids into train_users and test_users based on test_size
    
    # TODO: Aggregate all sequences from train_users into a single list (train_sequences)
    
    # TODO: Aggregate all sequences from test_users into a single list (test_sequences)
    
    pass

