"""
===============================================================
PAN estimator as Outlier Detector vs Classifier
===============================================================
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

# train the PAN estimator
model = ParallelAnomalousNudge.with_derived_estimators(X_train, y_train, random_seed=RANDOM_SEED, verbose=VERBOSE)
model.fit(X_train, y_train)

#%%

# inspect the labels yield by the default estimator type (Outlier detector)
y_pred = model.predict(X_test)
print("Unique labels of PAN predictions (Outlier detector):\t", np.unique(y_pred))
print("Prediction samples:\t", y_pred[:5])

#%% inspect the labels yield by the Classifier
y_pred = model.wrapAsClassifier().predict(X_test)
print("Unique labels of PAN predictions (Classifier):\t", np.unique(y_pred))
print("Prediction samples:\t", y_pred[:5])
