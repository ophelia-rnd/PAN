import numpy as np

from sklearn.base import BaseEstimator, check_is_fitted
from sklearn.preprocessing import StandardScaler
from sklearn.utils.multiclass import unique_labels
from ._somd import SomDetector
from .utils.som_hyparams import get_som_hyparams

class ParallelAnomalousNudge(BaseEstimator):
    """
    Parallel Anomalous Nudge (PAN) for detecting novelties.
    """

    def __init__(self, d1=4, d2=4, sigma=1.0, topology="rectangular", learning_rate=0.5, num_iteration=20,
                    decay_function="linear_decay_to_zero", sigma_decay_function="asymptotic_decay",
                    use_epochs=True, random_order=True, random_seed=None, verbose=False):

        self.d1 = d1
        self.d2 = d2
        self.sigma = sigma
        self.topology = topology
        self.learning_rate = learning_rate
        self.num_iteration = num_iteration
        self.decay_function = decay_function
        self.sigma_decay_function = sigma_decay_function
        self.use_epochs = use_epochs
        self.random_order = random_order
        self.random_seed = random_seed
        self.verbose = verbose

    def fit(self, X, y, continuity_violation_limit=0.1):

        X, y = self._validate_data(X, y)

        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y

        # Partition X by y
        X_partitions = {c: X[y == c] for c in self.classes_}
        self.X_partitions_ = X_partitions

        # Fit partition-wise scalers, transform X
        scalers = {c: StandardScaler().fit(X_partitions[c]) for c in self.classes_}
        X_partitions_scaled = {c: scalers[c].transform(X_partitions[c]) for c in self.classes_}
        self.scalers_ = scalers
        self.X_partitions_scaled_ = X_partitions_scaled

        # Learn partition-wise SOM representations
        self.estimators_: list[SomDetector] = []
        for c in self.classes_:
            XP_scaled = X_partitions_scaled[c]
            hyparams = get_som_hyparams(XP_scaled, verbose=self.verbose)
            estimator = SomDetector(**hyparams, random_seed=self.random_seed, verbose=self.verbose)
            estimator.fit(XP_scaled)
            self.estimators_.append(estimator)

        # Calculate normal train data scores and initialize offset
        X_combined_scores = self.combined_score_samples(X_partitions[0])
        self.offset_ = min(X_combined_scores)


        # TODO: assert continuity

        return self

    def score_samples(self, X):
        """
        Opposite of the deviation of X measured from the closest reference point (best-matching unit, BMU) of the trained SOM representation, for all representation, in a tuple format.
        The bigger is better, i.e. zero being the maximum value a sample can score, the closer the score is to zero, the more it is considered as an inlier.
        """

        check_is_fitted(self)
        X = self._validate_data(X)

        # Scale X
        X_scaled = np.array([self.scalers_[c].transform(X) for c in self.classes_]).T

        # Obtain representation-wise scores
        X_scores = np.array([est.score_samples(scaled_data) for est, scaled_data in zip(self.estimators_, X_scaled.T)]).T

        return X_scores

    def combined_score_samples(self, X):
        """
        Combine constituent score components of X resulted from `.score_samples(X)`.
        """

        check_is_fitted(self)
        X = self._validate_data(X)

        scores = self.score_samples(X)

        anomaly_weight = 1.0
        normalcy_scores = scores[:, 0]
        anomaloussness_scores = scores[:, 1]

        normalcy_scores *= -1
        anomaloussness_scores *= -1

        combines_scores = np.maximum(1, (anomaly_weight * normalcy_scores) / (anomaloussness_scores + 1e-12)) * normalcy_scores
        combines_scores *= -1

        return combines_scores

    def decision_function(self, X):
        check_is_fitted(self)
        X = self._validate_data(X, reset=False)

        return (self.combined_score_samples(X) - self.offset_)

    def predict(self, X):
        return (self.decision_function(X) < 0).astype(int)

