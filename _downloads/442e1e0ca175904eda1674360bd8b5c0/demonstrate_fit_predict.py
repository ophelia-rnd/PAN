"""
=====================================================
Demonstration of fitting the estimator and predicting
=====================================================

The following is a simple overview of how the :class:`pan.pan.ParallelAnomalousNudge` can be fitted and used as an estimator.
"""

# %%

import numpy as np

N_NORMAL = 100
N_ABNORMAL = 20

def generate_train_data(random_seed):
    np.random.seed(random_seed)
    # Generate normal samples
    f1 = np.random.normal(loc=.06, scale=.002, size=N_NORMAL)
    f2 = np.random.normal(loc=.075, scale=.002, size=N_NORMAL)
    X_normal = np.vstack((f1, f2)).T
    # Generate abnormal samples
    f1_abn = np.random.normal(loc=.06, scale=.002, size=N_ABNORMAL) + np.random.uniform(0.005, 0.01, size=N_ABNORMAL)
    f2_abn = np.random.normal(loc=.075, scale=.002, size=N_ABNORMAL) + np.random.uniform(0.005, 0.01, size=N_ABNORMAL)
    X_abnormal = np.vstack((f1_abn, f2_abn)).T
    # Combine
    X_train = np.vstack((X_normal, X_abnormal))
    y_train = np.hstack((np.zeros(N_NORMAL), np.ones(N_ABNORMAL)))
    return X_train, y_train

def generate_test_data(random_seed):
    np.random.seed(random_seed)
    # Generate normal samples
    f1 = np.random.normal(loc=.06, scale=.002, size=N_NORMAL)
    f2 = np.random.normal(loc=.075, scale=.002, size=N_NORMAL)
    X_normal = np.vstack((f1, f2)).T
    # Generate abnormal samples
    f1_abn = np.random.uniform(.06-0.015, .06+0.015, size=N_ABNORMAL)
    f2_abn = np.random.uniform(.075-0.015, .075+0.015, size=N_ABNORMAL)
    X_abnormal = np.vstack((f1_abn, f2_abn)).T
    # Combine
    X_test = np.vstack((X_normal, X_abnormal))
    y_test = np.hstack((np.zeros(N_NORMAL), np.ones(N_ABNORMAL)))
    return X_test, y_test

# %%

from pan import ParallelAnomalousNudge

# Generate train / test data
X_train, y_train = generate_train_data(random_seed=42)
X_test, y_test = generate_train_data(random_seed=84)

# Fit PAN model on train data
from sklearn.preprocessing import StandardScaler
estimator = ParallelAnomalousNudge(scaler=StandardScaler(), random_seed=42, verbose=True)
estimator.fit(X_train, y_train)

# %%

# Constituent score components
#   Column 1: Inlierness to learned normal representation, the closer to zero, the more inlier the sample is
#   Column 2: Inlierness to learned anomalous representation, the closer to zero, the more inlier the sample is

# Train samples
score_components_train = estimator.score_samples(X_train)

# Test samples
score_components_test = estimator.score_samples(X_test)

# %%

# Combined score components

# Train samples
combined_scores_train = estimator.combined_score_samples(X_train)

# Test samples
combined_scores_test = estimator.combined_score_samples(X_test)
