"""
===============================================================
Training the PAN estimator
===============================================================
"""

#%%
RANDOM_SEED = 42
VERBOSE = True

#%%

# generate train / test data
from examples.utils.dataset import generate_train_data, generate_test_data
X_train, y_train = generate_train_data(n_normal=100, n_abnormal=20, random_seed=RANDOM_SEED)
X_test, y_test = generate_test_data(n_normal=100, n_abnormal=20, random_seed=RANDOM_SEED*2)

#%%
from pan import ParallelAnomalousNudge, SomRepresentationEstimator
from minisom_representation import SomRepresentation
from sklearn.preprocessing import StandardScaler

#%%

# train with default settings aka. derived SOM representation estimators
model = ParallelAnomalousNudge.with_derived_estimators(X_train, y_train, random_seed=RANDOM_SEED, verbose=VERBOSE)
model.fit(X_train, y_train)

#%%

# train with SOM representation estimator hyperparameters specified
X_train_scaled = StandardScaler().fit_transform(X_train)
estimator_0 = SomRepresentationEstimator.with_derived_som_representation(X=X_train_scaled[y_train == 0], nu=.05, random_seed=RANDOM_SEED, verbose=VERBOSE)
estimator_1 = SomRepresentationEstimator.with_derived_som_representation(X=X_train_scaled[y_train == 1], nu=.05, random_seed=RANDOM_SEED, verbose=VERBOSE)
estimators = {0: estimator_0, 1: estimator_1}
model = ParallelAnomalousNudge.from_estimators(estimators=estimators, scaler=StandardScaler(), nu=estimator_0.nu, omega=3.5, random_seed=RANDOM_SEED, verbose=VERBOSE)
model.fit(X_train, y_train)

#%%

# train with SOM representation estimator and underlying SOM hyperparameters specified
X_train_scaled = StandardScaler().fit_transform(X_train)
som_rep_0 = SomRepresentation.with_derived_params(X=X_train_scaled[y_train == 0], sigma=2.5, learning_rate=0.25, num_iteration=25, random_seed=RANDOM_SEED, verbose=VERBOSE)
som_rep_1 = SomRepresentation.with_derived_params(X=X_train_scaled[y_train == 1], sigma=1.0, learning_rate=0.10, num_iteration=10, random_seed=RANDOM_SEED, verbose=VERBOSE)
estimator_0 = SomRepresentationEstimator.from_som_representation(som_rep_0, nu=0.25, random_seed=RANDOM_SEED, verbose=VERBOSE)
estimator_1 = SomRepresentationEstimator.from_som_representation(som_rep_1, nu=0.1, random_seed=RANDOM_SEED, verbose=VERBOSE)
estimators = {0: estimator_0, 1: estimator_1}
model = ParallelAnomalousNudge.from_estimators(estimators=estimators, scaler=StandardScaler(), nu=estimator_0.nu, omega=3.5, random_seed=RANDOM_SEED, verbose=VERBOSE)
model.fit(X_train, y_train)
