"""
=====================================================
Demonstration of fitting the estimator
=====================================================

The following is a simple overview of how the :class:`pan.pan.ParallelAnomalousNudge` can be fitted.
"""

# %%

import numpy as np
from utils.dataset import generate_train_data, generate_test_data

# %%

from pan import ParallelAnomalousNudge

# Generate train / test data
X_train, y_train = generate_train_data(n_normal=100, n_abnormal=20, random_seed=42)
X_test, y_test = generate_train_data(n_normal=100, n_abnormal=20, random_seed=84)

# Fit PAN model on train data
from sklearn.preprocessing import StandardScaler
estimator = ParallelAnomalousNudge(scaler=StandardScaler(), random_seed=42, verbose=True)
estimator.fit(X_train, y_train)
