import numpy as np

from sklearn.base import BaseEstimator, check_is_fitted, clone
from sklearn.preprocessing import StandardScaler
from sklearn.utils.multiclass import unique_labels
from ._somd import SomDetector
from .utils.som_hyparams import get_som_hyparams

class ParallelAnomalousNudge(BaseEstimator):
    """
    Parallel Anomalous Nudge (PAN) for detecting novelties.
    """

    def __init__(self, scaler=StandardScaler(), normal_label=0, abnormal_label=1, omega=2.0, random_seed=None, verbose=False):

        self.scaler = scaler
        self.normal_label = normal_label
        self.abnormal_label = abnormal_label
        self.omega = omega
        self.random_seed = random_seed
        self.verbose = verbose

    def fit(self, X, y, continuity_violation_limit=0.1):

        X, y = self._validate_data(X, y)

        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        self.normal_label_idx_ = np.argwhere(self.classes_ == self.normal_label)
        self.abnormal_label_idx_ = np.argwhere(self.classes_ == self.abnormal_label)

        # Partition X by y
        X_partitions = {c: X[y == c] for c in self.classes_}
        self.X_partitions_ = X_partitions

        # Fit partition-wise scalers, transform X
        scalers = {c: clone(self.scaler).fit(X_partitions[c]) for c in self.classes_}
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

        X_abnormal = X_partitions[self.abnormal_label]
        self.X_abnormal_sample_n_ = len(X_abnormal)
        self.X_abnormal_deviations_ranked_ = sorted(abs(self.score_samples(X_abnormal)[:, self.abnormal_label_idx_].ravel()))

        # FIXME: self.offset_
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
    
    def score_samples_without_nudge(self, X):
        """
        Opposite of the deviation of X measured from the closest reference point (best-matching unit, BMU) of the Normal SOM representation.
        The bigger is better, i.e. zero being the maximum value a sample can score, the closer the score is to zero, the more it is considered as an inlier.
        Ignores learned anomalous evidence.
        """
        check_is_fitted(self)
        X = self._validate_data(X)

        X_score = self.score_samples(X)
        return X_score[:, self.normal_label_idx_].ravel()

    def score_samples_nudged(self, X):
        """
        Opposite of the deviation of X measured from the closest reference point (best-matching unit, BMU) of the Normal SOM representation,
        boosted by a nudge factor based on deviation from the Abnormal representation.
        The nudge is determined by how samples rank in historical abnormal samples' deviation from the abnormal representation.
        The bigger is better, i.e. zero being the maximum value a sample can score, the closer the score is to zero, the more it is considered as an inlier.
        """
        check_is_fitted(self)
        X = self._validate_data(X)

        X_score = self.score_samples(X)
        X_normal_score = X_score[:, self.normal_label_idx_].ravel()
        X_abnormal_score = X_score[:, self.abnormal_label_idx_].ravel()

        X_anomalous_rank = np.searchsorted(self.X_abnormal_deviations_ranked_, abs(X_abnormal_score)) + 1
        X_nudged_score = self.__internal_scoring_formula(X_normal_score, X_anomalous_rank)

        return np.array(X_nudged_score)

    def __sample_nudge_amount(self, X_rank):
        multiplier = ((self.X_abnormal_sample_n_ + 1) - X_rank) / self.X_abnormal_sample_n_
        return (multiplier * (self.omega - 1)) + 1

    def __internal_scoring_formula(self, X_normal_score, X_anomalous_rank):
        return X_normal_score * self.__sample_nudge_amount(X_anomalous_rank)
