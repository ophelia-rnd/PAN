"""
===============================================================================
Demonstration of the nudge effect
===============================================================================
"""

# %%

import numpy as np
import matplotlib.pyplot as plt
from utils.dataset import generate_train_data, generate_test_data

# %%

from pan import ParallelAnomalousNudge

# Generate train / test data
X_train, y_train = generate_train_data(n_normal=100, n_abnormal=20, random_seed=42)
X_test, y_test = generate_train_data(n_normal=100, n_abnormal=20, random_seed=84)

# Fit PAN model on train data
from sklearn.preprocessing import StandardScaler
estimator = ParallelAnomalousNudge(scaler=StandardScaler(), nu=.05, omega=2.0, random_seed=42, verbose=False)
estimator.fit(X_train, y_train)

# %%

fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, sharey=True)
axes = axes.flatten()

# Baseline: Obtain nudge-free scores (anomaly scores with respect to the learned normal representation)
scores = estimator._score_component_normal(X_test)

ax = axes[0]
ax.set_title(r"Anomaly score based on Normal representation only ($0 \simeq \mathrm{inlier}$)", loc="left")
ax.violinplot(scores, positions=[1], vert=False)
ax.scatter(scores[y_test == 0], y=np.ones_like(scores[y_test == 0]), marker="o", edgecolor="k", color="none", s=400, alpha=.75, label="True normal", zorder=2)
ax.scatter(scores[y_test == 1], y=np.ones_like(scores[y_test == 1]), marker="x", color="crimson", s=200, alpha=.75, label="True anomaly", zorder=3)
ax.set_yticks([])

# Nudged scores
scores = estimator.score_samples(X_test)

ax = axes[1]
ax.set_title(r"Nudged anomaly score ($0 \simeq \mathrm{inlier}$)", loc="left")
ax.violinplot(scores, positions=[1], vert=False)
ax.scatter(scores[y_test == 0], y=np.ones_like(scores[y_test == 0]), marker="o", edgecolor="k", color="none", s=400, alpha=.75, label="True normal", zorder=2)
ax.scatter(scores[y_test == 1], y=np.ones_like(scores[y_test == 1]), marker="x", color="crimson", s=200, alpha=.75, label="True anomaly", zorder=3)
ax.set_yticks([])

plt.legend()
plt.tight_layout()
plt.show()

# %%
# Next, call `decision_function`, which turns raw scores into a shifted score, where 0 acts as the decision boundary.
# Also, visualize predictions via the `predict` method.

fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, sharey=True)
axes = axes.flatten()

shifted_scores = estimator.decision_function(X_test)

ax = axes[0]
ax.set_title(rf"Shifted nudged anomaly score ($0 < \mathrm{{outlier}}$, $\mathrm{{nu}}={estimator.nu}$, $\mathrm{{offset}}={round(estimator.offset_, 3)}$)", loc="left")
ax.violinplot(shifted_scores, positions=[1], vert=False)
ax.scatter(shifted_scores[y_test == 0], y=np.ones_like(shifted_scores[y_test == 0]), marker="o", edgecolor="k", color="none", s=400, alpha=.75, label="True normal", zorder=2)
ax.scatter(shifted_scores[y_test == 1], y=np.ones_like(shifted_scores[y_test == 1]), marker="x", color="crimson", s=200, alpha=.75, label="True anomaly", zorder=3)
ax.axvline(x=0, color="k", linestyle="dashed")
ax.set_yticks([])

predictions = estimator.predict(X_test)

ax = axes[1]
ax.set_title(rf"Predictions", loc="left")
ax.violinplot(shifted_scores, positions=[1], vert=False)
ax.scatter(shifted_scores[predictions == 1], y=np.ones_like(shifted_scores[predictions == 1]), marker="^", edgecolor="k", color="none", s=200, alpha=.75, label="Predicted normal", zorder=2)
ax.scatter(shifted_scores[predictions == -1], y=np.ones_like(shifted_scores[predictions == -1]), marker="^", edgecolor="crimson", color="none", s=200, alpha=.75, label="Predicted anomaly", zorder=3)
ax.axvline(x=0, color="k", linestyle="dashed")
ax.set_yticks([])

plt.legend()
plt.tight_layout()
plt.show()
