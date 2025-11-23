# HMM Trip Purpose Scaffolding Plan

This plan sets up the Python project structure and file templates for the CSE250A Final Project. The focus is on creating clear function headers, class structures, and detailed docstrings to guide the implementation of the HMM and baselines.

## 1. Project Structure Setup

- Create `src/` directory to house the source code.
- Files to create:
- `src/preprocessing.py`: Data loading and sequence generation.
- `src/hmm.py`: The custom Hidden Markov Model implementation.
- `src/baselines.py`: Rule-based and frequency-based models.
- `src/utils.py`: Evaluation metrics and indexing helpers.
- `main.py`: Driver script to run the experiment.

## 2. Implementation Details

### A. Data Preprocessing (`src/preprocessing.py`)

- **Functionality**:
- Define `MODE_MAPPING` and `PURPOSE_MAPPING` dictionaries.
- `load_and_process_data(filepath)`: Load CSV, apply mappings, sort by User -> Date -> Sequence.
- `create_user_sequences(df)`: Group data into sequences of (Mode, Purpose) tuples per user/day.
- `train_test_split_by_user(sequences, test_size=0.2)`: Split data ensuring all trips from a specific user go into the same split.

### B. Custom HMM (`src/hmm.py`)

- **Class**: `HiddenMarkovModel`
- **Methods to Scaffold**:
- `__init__(self, num_states, num_observations)`: Initialize storage for A, B, pi matrices.
- `initialize_parameters(self)`: Random or uniform initialization.
- `_forward(self, observation_sequence)`: Compute alpha values (forward probabilities).
- `_backward(self, observation_sequence)`: Compute beta values (backward probabilities).
- `fit_baum_welch(self, sequences, n_iter)`: Implement EM algorithm to update A, B, pi.
- `predict_viterbi(self, observation_sequence)`: Decode most likely hidden state sequence.
- **Note**: Will include detailed comments explaining the math (Alpha/Beta recursion, Gamma/Xi calculation) without writing the full logic.

### C. Baselines (`src/baselines.py`)

- **Class**: `TimeOfDayBaseline`
- Method `predict(timestamp)`: Returns purpose based on hour of day rules (7-10 AM -> work, etc.).
- **Class**: `FrequencyBaseline`
- Method `fit(train_data)`: Learn distribution of purposes given a mode.
- Method `predict(mode)`: Randomly sample purpose based on observed frequencies.

### D. Utilities (`src/utils.py`)

- `calculate_accuracy(true_sequences, predicted_sequences)`: Compare ground truth vs predictions.
- `LabelEncoder`: Simple helper to convert string labels (e.g., "car", "work") to integer indices (0, 1...) for matrix operations.

### E. Main Driver (`main.py`)

- Import all modules.
- Load data.
- Run split.
- Initialize and train HMM (placeholder call).
- Run evaluation on Test set.
- Run and evaluate baselines.