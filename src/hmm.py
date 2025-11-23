import numpy as np
from typing import List, Tuple

class HiddenMarkovModel:
    """
    Custom implementation of a Hidden Markov Model.
    
    Attributes:
        num_states (int): Number of hidden states (purposes).
        num_observations (int): Number of possible observations (modes).
        A (np.ndarray): Transition probability matrix (num_states x num_states).
                        A[i, j] = P(state_t+1 = j | state_t = i)
        B (np.ndarray): Emission probability matrix (num_states x num_observations).
                        B[i, k] = P(observation_t = k | state_t = i)
        pi (np.ndarray): Initial state probability distribution (num_states,).
                        pi[i] = P(state_0 = i)
    """
    
    def __init__(self, num_states: int, num_observations: int):
        self.num_states = num_states
        self.num_observations = num_observations
        
        self.A = None
        self.B = None
        self.pi = None
        
    def initialize_parameters(self, random_state: int = 42):
        """
        Initialize A, B, and pi with random probabilities (normalized).
        """
        np.random.seed(random_state)
        
        # TODO: Initialize self.A (random, normalize rows to sum to 1)
        
        # TODO: Initialize self.B (random, normalize rows to sum to 1)
        
        # TODO: Initialize self.pi (random, normalize to sum to 1)
        
        pass

    def _forward(self, observation_sequence: List[int]) -> np.ndarray:
        """
        Compute the alpha values (forward probabilities).
        
        alpha[t, i] = P(o_1, ..., o_t, q_t = i | lambda)
        
        Args:
            observation_sequence: List of integer indices representing observations.
            
        Returns:
            alpha: Matrix of shape (T, num_states).
        """
        T = len(observation_sequence)
        alpha = np.zeros((T, self.num_states))
        
        # TODO: Initialization (t=0)
        # alpha[0, i] = pi[i] * B[i, O_0]
        
        # TODO: Induction (t=1 to T-1)
        # alpha[t, j] = sum_i(alpha[t-1, i] * A[i, j]) * B[j, O_t]
        
        return alpha

    def _backward(self, observation_sequence: List[int]) -> np.ndarray:
        """
        Compute the beta values (backward probabilities).
        
        beta[t, i] = P(o_t+1, ..., o_T | q_t = i, lambda)
        
        Args:
            observation_sequence: List of integer indices representing observations.
            
        Returns:
            beta: Matrix of shape (T, num_states).
        """
        T = len(observation_sequence)
        beta = np.zeros((T, self.num_states))
        
        # TODO: Initialization (t=T-1)
        # beta[T-1, i] = 1
        
        # TODO: Induction (t=T-2 to 0)
        # beta[t, i] = sum_j(A[i, j] * B[j, O_t+1] * beta[t+1, j])
        
        return beta

    def fit_baum_welch(self, sequences: List[List[int]], n_iter: int = 10):
        """
        Train the HMM parameters using the Baum-Welch algorithm (EM).
        
        Args:
            sequences: List of observation sequences (each is a list of ints).
            n_iter: Number of EM iterations.
        """
        for iteration in range(n_iter):
            # Initialize accumulators for A_numer, A_denom, B_numer, B_denom, pi_accum
            
            for seq in sequences:
                # 1. E-Step: Compute forward (alpha) and backward (beta) probabilities
                # alpha = self._forward(seq)
                # beta = self._backward(seq)
                
                # Compute gamma (posterior probability of being in state i at time t)
                # gamma[t, i] = (alpha[t, i] * beta[t, i]) / P(O | lambda)
                
                # Compute xi (joint posterior of transitioning from i to j at time t)
                # xi[t, i, j] = (alpha[t, i] * A[i, j] * B[j, O_t+1] * beta[t+1, j]) / P(O | lambda)
                
                # Accumulate sufficient statistics for parameter updates
                pass
                
            # 2. M-Step: Update A, B, pi using accumulated values
            # self.A = ...
            # self.B = ...
            # self.pi = ...
            
            pass

    def predict_viterbi(self, observation_sequence: List[int]) -> List[int]:
        """
        Decode the most likely sequence of hidden states using the Viterbi algorithm.
        
        Args:
            observation_sequence: List of integer indices representing observations.
            
        Returns:
            List[int]: Most likely sequence of state indices.
        """
        T = len(observation_sequence)
        delta = np.zeros((T, self.num_states))
        psi = np.zeros((T, self.num_states), dtype=int)
        
        # TODO: Initialization
        # delta[0, i] = pi[i] * B[i, O_0]
        
        # TODO: Recursion
        # delta[t, j] = max_i(delta[t-1, i] * A[i, j]) * B[j, O_t]
        # psi[t, j] = argmax_i(delta[t-1, i] * A[i, j])
        
        # TODO: Termination
        # best_path_prob = max(delta[T-1, :])
        # last_state = argmax(delta[T-1, :])
        
        # TODO: Path Backtracking
        
        pass

