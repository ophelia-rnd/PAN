"""
==========================================================================
Observe proximity to learned representations and relationship to threshold
==========================================================================

"""

# %%

import numpy as np
from utils.dataset import generate_train_data, generate_test_data

# %%

from pan import ParallelAnomalousNudge

import sklearn

# Generate train / test data
X_train, y_train = generate_train_data(n_normal=100, n_abnormal=20, random_seed=42)
X_test, y_test = generate_test_data(n_normal=100, n_abnormal=20, random_seed=84)

# Fit PAN model on train data
from sklearn.preprocessing import StandardScaler
estimator = ParallelAnomalousNudge(scaler=StandardScaler(), nu=.01, omega=2.0, random_seed=42, verbose=True)
estimator.fit(X_train, y_train)

# %%
# Display score components and how they relate to the threshold based on the combined score.

import matplotlib.pyplot as plt
from pan import ScoreComponentDisplay

disp = ScoreComponentDisplay.from_estimator(estimator, np.vstack((X_train, X_test)),
                                            threshold_style={"colors": "black", "linewidths": 2, "linestyles": "dashed"},
                                            cmap="tab10", alpha=.25) \
    .plot_samples(X_train[y_train == 0], style_preset="train") \
    .plot_samples(X_train[y_train == 1], style_preset="train_anomaly") \
    .plot_samples(X_test[y_test == 0], style_preset="test") \
    .plot_samples(X_test[y_test == 1], style_preset="test_anomaly")

plt.margins(.1)
plt.legend()
plt.show()
