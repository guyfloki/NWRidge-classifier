import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

def generate_multiclass_data(n_samples=100, centers=5, random_state=42, test_size=0.3, stratify=None):
    """
    Generates multiclass data using make_blobs from sklearn and splits it into training and testing datasets.
    
    Parameters:
    - n_samples: Total number of samples to generate.
    - centers: Number of classes or list of class centers.
    - random_state: Random seed for reproducibility.
    - test_size: Proportion of data to use for testing.
    - stratify: Array for stratified sampling.
    
    Returns:
    - X_train: Training data features.
    - X_test: Testing data features.
    - y_train: Training data labels.
    - y_test: Testing data labels.
    """
    X, y = make_blobs(n_samples=n_samples, centers=centers, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)
    return X_train, X_test, y_train, y_test
