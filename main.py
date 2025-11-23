import pandas as pd
import numpy as np
from src.preprocessing import load_and_process_data, create_user_sequences, train_test_split_by_user
from src.hmm import HiddenMarkovModel
from src.baselines import TimeOfDayBaseline, FrequencyBaseline
from src.utils import LabelEncoder, calculate_accuracy

def main():
    # 1. Load and Preprocess Data
    print("Loading and processing data...")
    data_path = "250A_project/data/mode_purpose_hmm.csv" # Adjust path if necessary
    # df = load_and_process_data(data_path)
    
    # 2. Generate Sequences
    print("Generating user sequences...")
    # user_sequences = create_user_sequences(df)
    
    # 3. Train/Test Split
    print("Splitting data...")
    # train_seqs, test_seqs = train_test_split_by_user(user_sequences, test_size=0.2)
    
    # Prepare Encoders
    # mode_encoder = LabelEncoder()
    # purpose_encoder = LabelEncoder()
    
    # TODO: Fit encoders on all data (or just train data + known labels)
    # all_modes = ...
    # all_purposes = ...
    # mode_encoder.fit(all_modes)
    # purpose_encoder.fit(all_purposes)
    
    # Convert sequences to integers for HMM
    # train_seqs_indices = [[(mode_encoder.transform([m])[0], purpose_encoder.transform([p])[0]) for m, p in seq] for seq in train_seqs]
    # test_seqs_indices = ...
    
    # Extract just the observations (modes) and hidden states (purposes)
    # train_obs = [[x[0] for x in seq] for seq in train_seqs_indices]
    # train_states = [[x[1] for x in seq] for seq in train_seqs_indices]
    
    # 4. Train HMM
    print("Training HMM...")
    # num_states = len(purpose_encoder)
    # num_observations = len(mode_encoder)
    # hmm = HiddenMarkovModel(num_states, num_observations)
    # hmm.initialize_parameters()
    # hmm.fit_baum_welch(train_obs, n_iter=10)
    
    # 5. Evaluate HMM
    print("Evaluating HMM...")
    # test_obs = ...
    # test_states = ...
    # predicted_states_indices = [hmm.predict_viterbi(seq) for seq in test_obs]
    # predicted_purposes = [purpose_encoder.inverse_transform(seq) for seq in predicted_states_indices]
    # true_purposes = [purpose_encoder.inverse_transform(seq) for seq in test_states]
    
    # accuracy = calculate_accuracy(true_purposes, predicted_purposes)
    # print(f"HMM Accuracy: {accuracy:.4f}")
    
    # 6. Run Baselines
    print("Running Baselines...")
    
    # Frequency Baseline
    # freq_baseline = FrequencyBaseline()
    # freq_baseline.fit(train_seqs) # Note: Passes original (mode, purpose) tuples
    # freq_preds = ...
    # print(f"Frequency Baseline Accuracy: {freq_acc:.4f}")
    
    # Time of Day Baseline
    # tod_baseline = TimeOfDayBaseline()
    # tod_preds = ...
    # print(f"Time of Day Baseline Accuracy: {tod_acc:.4f}")

if __name__ == "__main__":
    main()

