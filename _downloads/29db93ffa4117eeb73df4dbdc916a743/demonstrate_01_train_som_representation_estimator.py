"""
===================================================================
Training the SOM representation estimator
===================================================================
"""

#%%
RANDOM_SEED = 42
VERBOSE = True

#%%

# generate train / test data
from examples.utils.dataset import generate_train_data
X_train, _ = generate_train_data(n_normal=100, n_abnormal=20, random_seed=RANDOM_SEED)

#%%
from pan import SomRepresentationEstimator
from minisom_representation import SomRepresentation

#%%

# train with derived SOM representation
estimator = SomRepresentationEstimator.with_derived_som_representation(X=X_train, random_seed=RANDOM_SEED, verbose=VERBOSE)
estimator.fit(X_train)

# %%

# train with SOM representation hyperparameters specified
som_rep = SomRepresentation.with_derived_params(
    X_train, sigma=3.3, topology="rectangular", learning_rate=0.25, num_iteration=44,
    decay_function="asymptotic_decay", sigma_decay_function="asymptotic_decay",
    neighborhood_function='gaussian', activation_distance='euclidean',
    random_seed=RANDOM_SEED, verbose=VERBOSE
)
estimator = SomRepresentationEstimator.from_som_representation(som_rep, nu=.01, random_seed=RANDOM_SEED, verbose=VERBOSE)
estimator.fit(X_train)
