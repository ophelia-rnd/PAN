"""
=====================================================
Demonstration of using PAN as a classifier
=====================================================

"""

# %%

import numpy as np
from utils.dataset import generate_train_data, generate_test_data

# %%

from pan import ParallelAnomalousNudge

# Generate train / test data
X_train, y_train = generate_train_data(n_normal=100, n_abnormal=20, random_seed=42)
X_test, y_test = generate_test_data(n_normal=100, n_abnormal=20, random_seed=84)

# Fit PAN model on train data
from sklearn.preprocessing import StandardScaler
estimator = ParallelAnomalousNudge(scaler=StandardScaler(), nu=.05, omega=2.0, random_seed=42, verbose=True)
estimator.fit(X_train, y_train)

# %%

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, sharey=True)
axes = axes.flatten()

decision_scores = estimator.decision_function(X_test)

ax = axes[0]
ax.set_title(rf"Decision scores ($\mathrm{{outlier}} < 0$)", loc="left")
ax.violinplot(decision_scores, positions=[1], vert=False)
ax.scatter(decision_scores[y_test == 0], y=np.ones_like(decision_scores[y_test == 0]), marker="o", edgecolor="k", color="none", s=400, alpha=.75, label="True normal", zorder=2)
ax.scatter(decision_scores[y_test == 1], y=np.ones_like(decision_scores[y_test == 1]), marker="x", color="crimson", s=200, alpha=.75, label="True anomaly", zorder=3)
ax.axvline(x=0, color="k", linestyle="dashed")
ax.set_yticks([])

classifier_decision_scores = estimator.wrapAsClassifier().decision_function(X_test)

ax = axes[1]
ax.set_title(rf"Classifier decision scores ($0 < \mathrm{{outlier}}$)", loc="left")
ax.violinplot(classifier_decision_scores, positions=[1], vert=False)
ax.scatter(classifier_decision_scores[y_test == 0], y=np.ones_like(classifier_decision_scores[y_test == 0]), marker="o", edgecolor="k", color="none", s=400, alpha=.75, label="True normal", zorder=2)
ax.scatter(classifier_decision_scores[y_test == 1], y=np.ones_like(classifier_decision_scores[y_test == 1]), marker="x", color="crimson", s=200, alpha=.75, label="True anomaly", zorder=3)
ax.axvline(x=0, color="k", linestyle="dashed")
ax.set_yticks([])

plt.legend()
plt.tight_layout()
plt.show()
