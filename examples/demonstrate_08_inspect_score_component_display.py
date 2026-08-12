"""
==================================================================
Proximity to learned representations and relationship to threshold
==================================================================
"""

#%%
RANDOM_SEED = 42
VERBOSE = False

#%%

# generate train / test data
from examples.utils.dataset import generate_train_data, generate_test_data
X_train, y_train = generate_train_data(n_normal=100, n_abnormal=20, random_seed=RANDOM_SEED)
X_test, y_test = generate_test_data(n_normal=100, n_abnormal=20, random_seed=RANDOM_SEED*2)

#%%
import numpy as np
from pan import ParallelAnomalousNudge

#%%

# train PAN
model = ParallelAnomalousNudge.with_derived_estimators(X_train, y_train, nu=.1, random_seed=RANDOM_SEED, verbose=VERBOSE)
model.fit(X_train, y_train)

# %%

import matplotlib.pyplot as plt
from pan import ScoreComponentDisplay

fig = plt.figure(figsize=(12, 12))

disp = ScoreComponentDisplay \
    .from_estimator(
        model, np.vstack((X_train, X_test)),
        ax=fig.gca(), threshold_style={"colors": "black", "linewidths": 2, "linestyles": "dashed"}
    ) \
    .plot_samples(X_train[y_train == 0], style_preset="train") \
    .plot_samples(X_train[y_train == 1], style_preset="train_anomaly") \
    .plot_samples(X_test[y_test == 0], style_preset="test") \
    .plot_samples(X_test[y_test == 1], style_preset="test_anomaly")

plt.margins(.1)
plt.legend()
plt.show()
