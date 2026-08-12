"""
=================================================================
Scoring samples based on deviation from the normal representation
=================================================================
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
import matplotlib.pyplot as plt
import numpy as np
from pan import ParallelAnomalousNudge

#%%

# train PAN
model = ParallelAnomalousNudge.with_derived_estimators(X_train, y_train, nu=.1, random_seed=RANDOM_SEED, verbose=VERBOSE)
model.fit(X_train, y_train)

# %%

scores = model._score_component_normal(X_test)

y_values = [np.ones_like(scores[y_test == 0]), np.ones_like(scores[y_test == 1])]
plt.figure(figsize=(8, 5))
plt.title(r"Anomaly score considering only the Normal representation", loc="left")
plt.text(0, .8, s="inlier", va="center", ha="center", fontsize=12, color="white", bbox=dict(color="seagreen", boxstyle='round,pad=.6'), zorder=4)
plt.text(min(scores), .8, s="outlier", va="center", ha="center", fontsize=12, color="white", bbox=dict(color="indianred", boxstyle='round,pad=.6'), zorder=4)
plt.violinplot(scores, positions=[1], vert=False)
plt.scatter(scores[y_test == 0], y=y_values[0], marker="o", edgecolor="k", color="none", s=400, alpha=.75, label="True normal", zorder=2)
plt.scatter(scores[y_test == 1], y=y_values[1], marker="x", color="crimson", s=200, alpha=.75, label="True anomaly", zorder=3)
plt.legend()
plt.margins(.1)
plt.yticks([])
plt.show()
