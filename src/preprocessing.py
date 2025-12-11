import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import random
from collections import Counter

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
    "home": "home",
    "work": "work",
    "eat": "eat",

    # leisure group
    "leisure": "leisure",
    "sport": "leisure",
    "family_friends": "leisure",
    "shopping": "leisure",

    # errand group
    "errand": "errand",
    "assistance": "errand",
    "wait": "errand",
}

# ==========================================
# Time Bin Helper Functions
# ==========================================

def get_time_of_day_bin(timestamp) -> str:
    """
    Return time-of-day bin name for a timestamp.
    
    Bins:
        - night: [00:00, 06:00)
        - morning: [06:00, 11:00)
        - afternoon: [11:00, 17:00)
        - evening: [17:00, 24:00)
    
    Args:
        timestamp: pandas Timestamp or datetime object with .hour attribute.
    
    Returns:
        str: One of "night", "morning", "afternoon", "evening".
    """
    hour = timestamp.hour
    if 0 <= hour < 6:
        return "night"
    elif 6 <= hour < 11:
        return "morning"
    elif 11 <= hour < 17:
        return "afternoon"
    else:
        return "evening"


def get_hour_bin(timestamp) -> str:
    """
    Return hour as a string label (e.g., "08h", "17h").
    
    Args:
        timestamp: pandas Timestamp or datetime object with .hour attribute.
    
    Returns:
        str: Hour formatted as "XXh" (e.g., "00h", "13h", "23h").
    """
    return f"{timestamp.hour:02d}h"


# ==========================================
# Observation Encoding Configurations
# ==========================================

OBSERVATION_CONFIGS = {
    "mode_only": {
        "time_bin_fn": None,
        "description": "Use transport mode only (e.g., 'car', 'walk')",
    },
    "mode_time_of_day": {
        "time_bin_fn": get_time_of_day_bin,
        "description": "Mode + 4 time bins (e.g., 'car_morning', 'walk_evening')",
    },
    "mode_hour": {
        "time_bin_fn": get_hour_bin,
        "description": "Mode + 24 hour bins (e.g., 'car_08h', 'walk_17h')",
    },
}


def get_observation_label(mode: str, timestamp, config_name: str = "mode_only") -> str:
    """
    Return observation label based on the specified configuration.
    
    Args:
        mode: Transport mode string (e.g., "car", "walk").
        timestamp: pandas Timestamp or datetime object.
        config_name: Key into OBSERVATION_CONFIGS. One of:
            - "mode_only": Returns mode as-is.
            - "mode_time_of_day": Returns "mode_timebin" (e.g., "car_morning").
            - "mode_hour": Returns "mode_hour" (e.g., "car_08h").
    
    Returns:
        str: Observation label for use with LabelEncoder.
    """
    config = OBSERVATION_CONFIGS[config_name]
    if config["time_bin_fn"] is None:
        return mode
    time_bin = config["time_bin_fn"](timestamp)
    return f"{mode}_{time_bin}"


def load_and_process_data(filepath: str) -> pd.DataFrame:
    """
    Load the dataset, apply mappings, and sort.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Processed dataframe with mapped 'mode' and 'purpose' columns,
                      sorted by user_id, event_date, and seq_idx.
    """
    # Load CSV
    df = pd.read_csv(filepath)
    
    # Apply MODE_MAPPING to 'trip_mode' column to create 'mode' column
    df['mode'] = df['trip_mode'].map(MODE_MAPPING)

    # Apply PURPOSE_MAPPING to 'end_purpose' column to create 'purpose' column
    df['purpose'] = df['end_purpose'].map(PURPOSE_MAPPING)

    
    # convert event date and event time to datetime
    
    df['event_date'] = pd.to_datetime(df['event_date'])
    df['event_time'] = pd.to_datetime(df['event_time'])
    # Sort by ['user_id', 'event_date', 'seq_idx']
    df = df.sort_values(by=['user_id','event_date','seq_idx'])
    # # Filter out any rows where mode or purpose is NaN after mapping (if any)
    
    df = df.dropna()
    
    return df

def create_user_sequences(df: pd.DataFrame, obs_config: str = "mode_only") -> Dict[str, List[List[Tuple[str, str, Any]]]]:
    """
    Group data into sequences of (observation_label, purpose, event_time) tuples per user.
    
    Args:
        df (pd.DataFrame): Sorted processed dataframe.
        obs_config (str): Observation encoding configuration. One of:
            - "mode_only": Use transport mode only (e.g., 'car', 'walk').
            - "mode_time_of_day": Mode + 4 time bins (e.g., 'car_morning', 'walk_evening').
            - "mode_hour": Mode + 24 hour bins (e.g., 'car_08h', 'walk_17h').

    Returns:
        Dict[str, List[List[Tuple[str, str, Any]]]]: A dictionary where keys are user_ids
        and values are lists of sequences.
        Each sequence is a list of (observation_label, purpose, event_time) tuples 
        corresponding to a day (or continuous block).
        
        Example structure (with obs_config="mode_time_of_day"):
        {
            'user_1': [
                [('car_evening', 'work', timestamp), ('walk_evening', 'eat', timestamp), ...],  # Day 1
                [('bus_morning', 'leisure', timestamp), ('bus_afternoon', 'home', timestamp)]   # Day 2
            ],
            ...
        }
    """
    user_dict = {}
    # Group by user_id
    user_trips = df.groupby('user_id')
    # NOTE: user_trips is a groupby object, not a df!
    for user_id, user_df in user_trips:
        user_dict[user_id] = []

        daily_trips = user_df.groupby('event_date')
        for _, daily_user_trip in daily_trips:
            seq = []
            for _, row in daily_user_trip.iterrows():
                obs_label = get_observation_label(row['mode'], row['event_time'], obs_config)
                seq.append((obs_label, row['purpose'], row['event_time']))
            user_dict[user_id].append(seq)

    return user_dict



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
    # all unique user_ids
    user_ids = list(sequences.keys())
    #  shuffle user_ids
    random.seed(random_seed)
    random.shuffle(user_ids)
    # Split user_ids into train_users and test_users based on test_size
    user_count = len(user_ids)
    test_count = int(user_count * test_size)
    test_user = user_ids[:test_count]
    train_user = user_ids[test_count:]
    # Agg all sequences from train_users into a single list (train_sequences)
    train_sequences = []
    for u in train_user:
        for seq in sequences[u]:
            train_sequences.append(seq)
    #Agg all sequences from test_users into a single list (test_sequences)
    test_sequences = []
    for u in test_user:
        for seq in sequences[u]:
            test_sequences.append(seq)
    
    return (train_sequences,test_sequences)


# ==========================================
# Visualization Functions
# ==========================================

# Base colors for each transport mode
MODE_COLORS = {
    "car": "#e74c3c",      # red
    "bus": "#3498db",      # blue
    "bike": "#2ecc71",     # green
    "train": "#f1c40f",    # yellow
    "walk": "#9b59b6",     # purple
}

# Time bin ordering for gradient (morning strongest -> night softest)
TIME_BIN_ALPHA = {
    "morning": 1.0,
    "afternoon": 0.75,
    "evening": 0.5,
    "night": 0.35,
}

# Time bin sort order (for proper gradient display)
TIME_BIN_ORDER = {"morning": 0, "afternoon": 1, "evening": 2, "night": 3}

# Display titles for observation configs
CONFIG_DISPLAY_TITLES = {
    "mode_only": "Mode Only",
    "mode_time_of_day": "Mode and Time of Day",
    "mode_hour": "Mode and Hour of Day",
}


def _get_bar_color(label: str, config_name: str):
    """
    Get the color for a bar based on the observation label and config.
    
    For mode_only: returns base mode color.
    For mode_time_of_day: returns base color with alpha gradient.
    For mode_hour: returns base color with hour-based alpha gradient.
    
    Returns:
        Tuple[str, float]: (hex_color, alpha)
    """
    if config_name == "mode_only":
        return MODE_COLORS.get(label, "#95a5a6"), 1.0
    
    # Parse mode from joint label (e.g., "car_morning" -> "car")
    parts = label.rsplit("_", 1)
    if len(parts) != 2:
        return "#95a5a6", 1.0
    
    mode, time_part = parts
    base_color = MODE_COLORS.get(mode, "#95a5a6")
    
    if config_name == "mode_time_of_day":
        alpha = TIME_BIN_ALPHA.get(time_part, 0.7)
        return base_color, alpha
    
    elif config_name == "mode_hour":
        # Parse hour (e.g., "08h" -> 8)
        try:
            hour = int(time_part.replace("h", ""))
            # Gradient: morning hours (6-11) strongest, night hours (0-5) softest
            # Map hour to alpha: 6am = 1.0, decreasing through day, 5am = 0.3
            if 6 <= hour < 12:
                alpha = 1.0 - (hour - 6) * 0.05  # 1.0 -> 0.7
            elif 12 <= hour < 18:
                alpha = 0.7 - (hour - 12) * 0.05  # 0.7 -> 0.4
            elif 18 <= hour < 24:
                alpha = 0.4 - (hour - 18) * 0.02  # 0.4 -> 0.28
            else:  # 0-5
                alpha = 0.3 + hour * 0.02  # 0.3 -> 0.4
            return base_color, max(0.25, alpha)
        except ValueError:
            return base_color, 0.7
    
    return base_color, 0.7


def plot_observation_distribution_comparison(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (18, 6),
):
    """
    Plot side-by-side comparison of all observation configs in a single row.
    
    Each mode has a unique color (car=red, bus=blue, bike=green, train=yellow, walk=purple).
    For time-based configs, gradients show time progression (morning=strong, night=soft).
    
    Args:
        df (pd.DataFrame): Processed dataframe from load_and_process_data().
        figsize (Tuple[int, int]): Figure size (width, height).
    
    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    obs_configs = ["mode_only", "mode_time_of_day", "mode_hour"]
    fig, axes = plt.subplots(1, len(obs_configs), figsize=figsize)
    
    for idx, config_name in enumerate(obs_configs):
        ax = axes[idx]
        
        # Generate observation labels
        obs_labels = [
            get_observation_label(mode, timestamp, config_name)
            for mode, timestamp in zip(df['mode'], df['event_time'])
        ]
        
        # Count and normalize
        counts = Counter(obs_labels)
        total = sum(counts.values())
        
        # Custom sorting based on config
        if config_name == "mode_time_of_day":
            # Sort by mode first, then by time order (morning -> afternoon -> evening -> night)
            def sort_key(label):
                parts = label.rsplit("_", 1)
                if len(parts) == 2:
                    mode, time_bin = parts
                    return (mode, TIME_BIN_ORDER.get(time_bin, 99))
                return (label, 0)
            sorted_labels = sorted(counts.keys(), key=sort_key)
        elif config_name == "mode_hour":
            # Sort by mode first, then by hour starting at 6am (strongest gradient)
            # Order: 6,7,8,...,23,0,1,2,3,4,5 (morning->afternoon->evening->night)
            def sort_key(label):
                parts = label.rsplit("_", 1)
                if len(parts) == 2:
                    mode, hour_str = parts
                    try:
                        hour = int(hour_str.replace("h", ""))
                        # Shift so 6am=0, 7am=1, ..., 5am=23
                        hour_order = (hour - 6) % 24
                        return (mode, hour_order)
                    except ValueError:
                        return (mode, 99)
                return (label, 0)
            sorted_labels = sorted(counts.keys(), key=sort_key)
        else:
            sorted_labels = sorted(counts.keys())
        
        proportions = [counts[label] / total for label in sorted_labels]
        
        # Get colors for each bar
        bar_colors = []
        bar_alphas = []
        for label in sorted_labels:
            color, alpha = _get_bar_color(label, config_name)
            bar_colors.append(color)
            bar_alphas.append(alpha)
        
        # Plot bars individually to apply different alphas
        for i, (label, prop, color, alpha) in enumerate(zip(sorted_labels, proportions, bar_colors, bar_alphas)):
            ax.bar(i, prop, color=color, alpha=alpha, edgecolor='black', linewidth=0.5)
        
        ax.set_xticks(range(len(sorted_labels)))
        
        # Adjust label display based on number of classes
        if len(sorted_labels) <= 20:
            ax.set_xticklabels(sorted_labels, rotation=45, ha='right', fontsize=8)
        else:
            # For many classes, skip labels
            ax.set_xticklabels(['' for _ in sorted_labels])
            ax.set_xlabel(f'{len(sorted_labels)} classes', fontsize=10)
        
        ax.set_ylabel('Proportion', fontsize=10)
        display_title = CONFIG_DISPLAY_TITLES.get(config_name, config_name)
        ax.set_title(f'{display_title}\n({len(sorted_labels)} classes)', fontsize=11)
        
        # Add summary stats
        max_prop = max(proportions)
        min_prop = min(proportions)
        ax.text(0.95, 0.95, f'max: {max_prop:.3f}\nmin: {min_prop:.3f}',
               transform=ax.transAxes, ha='right', va='top', fontsize=8,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add legend for mode colors in the first subplot (Mode Only), top left corner
    legend_patches = [mpatches.Patch(color=color, label=mode.capitalize()) 
                      for mode, color in MODE_COLORS.items()]
    axes[0].legend(handles=legend_patches, loc='upper left', fontsize=8, title='Mode',
                   title_fontsize=9, framealpha=0.9)
    
    fig.suptitle('Observation Distribution Comparison Across Encoding Strategies', 
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])  # Leave room for title only
    return fig


def plot_purpose_distribution(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (10, 5),
):
    """
    Plot normalized distribution of trip purposes.
    
    Bars are sorted by proportion (largest on left).
    
    Args:
        df (pd.DataFrame): Processed dataframe from load_and_process_data().
        figsize (Tuple[int, int]): Figure size (width, height).
    
    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=figsize)
    
    counts = Counter(df['purpose'])
    total = sum(counts.values())
    
    # Sort by proportion descending (largest bars on left)
    sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    sorted_labels = [item[0] for item in sorted_items]
    proportions = [item[1] / total for item in sorted_items]
    
    # Vibrant orange bars
    bars = ax.bar(range(len(sorted_labels)), proportions, 
                  color='#ff6b00', edgecolor='black', alpha=0.9, linewidth=1)
    
    ax.set_xticks(range(len(sorted_labels)))
    ax.set_xticklabels(sorted_labels, rotation=45, ha='right', fontsize=11)
    ax.set_ylabel('Proportion', fontsize=11)
    ax.set_xlabel('Purpose', fontsize=11)
    ax.set_title(f'Trip Purpose Distribution (n_classes={len(sorted_labels)})', 
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(proportions) * 1.15)
    
    # Add proportion labels on bars
    for bar, prop in zip(bars, proportions):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
               f'{prop:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    return fig


# TEST CASES

if __name__ == "__main__":
    # test

    filepath = "../data/mode_purpose_hmm.csv"

    df = load_and_process_data(filepath)

    # Plot observation distribution comparison (mode-colored with gradients)
    fig1 = plot_observation_distribution_comparison(df=df)
    fig1.savefig("observation_distribution_comparison.png", dpi=150, bbox_inches='tight')
    
    # Plot purpose distribution (orange bars)
    fig2 = plot_purpose_distribution(df=df)
    fig2.savefig("purpose_distribution.png", dpi=150, bbox_inches='tight')

    import pdb; pdb.set_trace()

    user_dict = create_user_sequences(df=df, obs_config="mode_time_of_day")

    # import pdb; pdb.set_trace()

    for user,trips in user_dict.items():
        print("------ user -------")
        print(user)
        print('------ trips of the user -----')
        print(trips)
        break

    tp = train_test_split_by_user(sequences=user_dict,test_size=0.2,random_seed=42)
    train, test = tp[0], tp[1]

    print("----- train set entry -----")
    print(train[0])
 
    # summary
    print("----- data summary -----")
    print("Train len:", len(train))
    print("Test len:", len(test))
    print("Sample train seq length:", len(train[2]))
    print("First few obs in first train seq:", train[2][:3])

    # stats
    max_len_train=max_len_test=0
    min_len_train=min_len_test=1000

    for seq_train,seq_test in zip(train,test):
        if len(seq_train)>max_len_train:
            max_len_train = len(seq_train)
        elif len(seq_train)<min_len_train:
            min_len_train = len(seq_train)

        if len(seq_test)>max_len_test:
            max_len_test = len(seq_test)
        elif len(seq_test)<min_len_test:
            min_len_test = len(seq_test)
    # train stats
    print("----- training set stats -----")
    print(f"maxixum length of daily trips in training: {max_len_train}")
    print(f"min length of daily trips in training: {min_len_train} ")
    print('----- testing set stats -----')
    print(f"maxixum length of daily trips in testing: {max_len_test}")
    print(f"min length of daily trips in testing: {min_len_test} ")
    




