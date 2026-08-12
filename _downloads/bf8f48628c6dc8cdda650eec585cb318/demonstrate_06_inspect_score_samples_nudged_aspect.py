"""
=================================================================
Scoring samples based on the nudge mechanism
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

fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True, sharey=True)
axes = axes.flatten()

scores = model._score_component_normal(X_test)
y_values = [np.ones_like(scores[y_test == 0]), np.ones_like(scores[y_test == 1])]

ax = axes[0]
ax.set_title(r"Anomaly score considering only the Normal representation", loc="left")
ax.text(0, .8, s="inlier", va="center", ha="center", fontsize=12, color="white", bbox=dict(color="seagreen", boxstyle='round,pad=.6'), zorder=4)
ax.text(min(scores), .8, s="outlier", va="center", ha="center", fontsize=12, color="white", bbox=dict(color="indianred", boxstyle='round,pad=.6'), zorder=4)
ax.violinplot(scores, positions=[1], vert=False)
ax.scatter(scores[y_test == 0], y=y_values[0], marker="o", edgecolor="k", color="none", s=400, alpha=.75, label="True normal", zorder=2)
ax.scatter(scores[y_test == 1], y=y_values[1], marker="x", color="crimson", s=200, alpha=.75, label="True anomaly", zorder=3)
ax.legend()
ax.margins(.1)

scores = model.score_samples(X_test)

ax = axes[1]
ax.set_title(r"Nudged anomaly score", loc="left")
ax.text(0, .8, s="inlier", va="center", ha="center", fontsize=12, color="white", bbox=dict(color="seagreen", boxstyle='round,pad=.6'), zorder=4)
ax.text(min(scores), .8, s="outlier", va="center", ha="center", fontsize=12, color="white", bbox=dict(color="indianred", boxstyle='round,pad=.6'), zorder=4)
ax.violinplot(scores, positions=[1], vert=False)
ax.scatter(scores[y_test == 0], y=y_values[0], marker="o", edgecolor="k", color="none", s=400, alpha=.75, label="True normal", zorder=2)
ax.scatter(scores[y_test == 1], y=y_values[1], marker="x", color="crimson", s=200, alpha=.75, label="True anomaly", zorder=3)
ax.legend()
ax.margins(.1)

plt.yticks([])
plt.show()

# %%

scores_0 = model._score_component_normal(X_test)
scores_n = model.score_samples(X_test)
nudged_inds = scores_n < scores_0
print(f"Number of nudged samples: {nudged_inds.sum()}")

#%%

y_values = [np.ones_like(scores[y_test == 0]), np.ones_like(scores[y_test == 1])]
plt.figure(figsize=(8, 5))
plt.title(r"Effect of nudge", loc="left")
plt.text(0, .8, s="inlier", va="center", ha="center", fontsize=12, color="white", bbox=dict(color="seagreen", boxstyle='round,pad=.6'), zorder=4)
plt.text(min(scores), .8, s="outlier", va="center", ha="center", fontsize=12, color="white", bbox=dict(color="indianred", boxstyle='round,pad=.6'), zorder=4)
plt.violinplot(scores, positions=[1], vert=False)
plt.scatter(scores_0[nudged_inds], y=np.ones_like(scores_0[nudged_inds]), marker=".", edgecolor="k", color="k", s=300, alpha=.4, zorder=2, label="Normalcy score")
plt.scatter(scores_n[nudged_inds], y=np.ones_like(scores_n[nudged_inds]), marker=7, color="purple", s=300, alpha=.5, zorder=3, label="Nudged score")
plt.legend()
plt.margins(.1)
plt.yticks([])
plt.show()
