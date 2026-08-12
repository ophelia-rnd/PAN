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

fig,axes = plt.subplots(1, 2, figsize=(8, 5), sharex=True, sharey=True)

ax = axes[0]
ax.set_title("Training data")
ax.scatter(*X_train.T, c=y_train, cmap="Spectral_r", alpha=.75)

ax = axes[1]
ax.set_title("Test data")
ax.scatter(*X_test.T, c=y_test, cmap="Spectral_r", alpha=.75)

for ax in axes:
    ax.set_aspect('equal')

plt.suptitle("Example dataset")
plt.tight_layout()
plt.show()
