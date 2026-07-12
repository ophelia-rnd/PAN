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
estimator = ParallelAnomalousNudge(scaler=StandardScaler(), nu=.05, omega=2.0, random_seed=42, verbose=True)
estimator.fit(X_train, y_train)

# %%
# The `ParallelAnomalousNudge` itself does not support setting the hyperparameters for the class-conditioned SOM representations that are being learned internally.
# Instead, one can pass blueprints of :class:`pan.pan.SomRepresentation` as parameters that can be configured for each class.

from pan import ParallelAnomalousNudge, SomRepresentation

som_estimator_normal = SomRepresentation(d1=10, d2=5, sigma=5.0, learning_rate=.6, num_iteration=5, verbose=True)
som_estimator_abnormal = SomRepresentation(d1=3, d2=3, sigma=2.5, learning_rate=.6, num_iteration=5, verbose=True)

som_estimators = {
    0: som_estimator_normal,
    1: som_estimator_abnormal
}

estimator = ParallelAnomalousNudge(estimators=som_estimators, scaler=StandardScaler(), nu=.05, omega=2.0, random_seed=42, verbose=True)
estimator.fit(X_train, y_train)
