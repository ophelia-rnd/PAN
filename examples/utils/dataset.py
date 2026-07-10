import numpy as np

def generate_train_data(n_normal=100, n_abnormal=20, random_seed=None):
    np.random.seed(random_seed)
    # Generate normal samples
    f1 = np.random.normal(loc=.06, scale=.002, size=n_normal)
    f2 = np.random.normal(loc=.075, scale=.002, size=n_normal)
    X_normal = np.vstack((f1, f2)).T
    # Generate abnormal samples
    f1_abn = np.random.normal(loc=.06, scale=.002, size=n_abnormal) + np.random.uniform(0.005, 0.01, size=n_abnormal)
    f2_abn = np.random.normal(loc=.075, scale=.002, size=n_abnormal) + np.random.uniform(0.005, 0.01, size=n_abnormal)
    X_abnormal = np.vstack((f1_abn, f2_abn)).T
    # Combine
    X_train = np.vstack((X_normal, X_abnormal))
    y_train = np.hstack((np.zeros(n_normal), np.ones(n_abnormal)))
    return X_train, y_train

def generate_test_data(n_normal=100, n_abnormal=20, random_seed=None):
    np.random.seed(random_seed)
    # Generate normal samples
    f1 = np.random.normal(loc=.06, scale=.002, size=n_normal)
    f2 = np.random.normal(loc=.075, scale=.002, size=n_normal)
    X_normal = np.vstack((f1, f2)).T
    # Generate abnormal samples
    f1_abn = np.random.uniform(.06-0.015, .06+0.015, size=n_abnormal)
    f2_abn = np.random.uniform(.075-0.015, .075+0.015, size=n_abnormal)
    X_abnormal = np.vstack((f1_abn, f2_abn)).T
    # Combine
    X_test = np.vstack((X_normal, X_abnormal))
    y_test = np.hstack((np.zeros(n_normal), np.ones(n_abnormal)))
    return X_test, y_test