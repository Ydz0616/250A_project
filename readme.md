# Trip Purpose Prediction

This repository contains the code and resources for predicting trip purposes using Hidden Markov Models (HMM) and other baseline methods.

## Overview

Understanding the purpose of trips is crucial for transportation planning and analysis. This project implements a machine learning pipeline to predict trip purposes based on sequence data. It includes data preprocessing, model implementation (specifically HMM), baseline comparisons, and visualization of results.

## Project Structure

The repository is organized as follows:

```
src/
├── hmm.py                  # Implementation of the Hidden Markov Model logic
├── baselines.py            # Baseline models for performance comparison
├── preprocessing.py        # Data cleaning, transformation, and sequence preparation
├── utils.py                # General utility and helper functions
├── *.png                   # Generated visualizations (e.g., purpose_distribution.png)
data/                        # Datasets directory
README_trip_sequence.md     # Documentation on trip sequence data format
latex/
├── main.tex                # LaTeX source for the research paper/report
plans/                      # Project planning and scaffolding documents
demo.ipynb                  # Jupyter Notebook demo of the full pipeline
demo.html                   # HTML export of the demo notebook
```

## Getting Started

### Prerequisites

To run the code, you will need Python installed along with common data science libraries. While a dedicated `requirements.txt` is not provided, the project relies on:

- numpy
- pandas
- matplotlib / seaborn
- scikit-learn
- jupyter

### Usage

#### Explore the Demo

The recommended starting point is the `demo.ipynb` notebook, which walks through data loading, HMM-based modeling, and result visualization.

```bash
jupyter notebook demo.ipynb
```

#### Data Processing

Refer to `src/preprocessing.py` for details on how raw trip data is transformed into sequential inputs suitable for HMM-based modeling.

#### Model Training

- **HMM models:** Implemented in `src/hmm.py`
- **Baselines:** Implemented in `src/baselines.py`

## Methodology

The primary approach leverages Hidden Markov Models (HMMs), treating trip purposes as latent states that generate observed trip characteristics. This formulation enables inference over entire trip chains rather than independent, point-wise predictions.

## Visualizations

The `src/` directory includes generated plots for exploratory analysis and evaluation:

- **Purpose distribution:** Frequency of different trip purposes
- **Observation distribution:** Comparison of feature distributions across categories

## References

For mathematical details, implementation specifics, and experimental results, please refer to the documents in the `latex/` directory.
