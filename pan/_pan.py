import numpy as np

from sklearn.base import BaseEstimator, check_is_fitted, clone
from sklearn.preprocessing import StandardScaler
from sklearn.utils.multiclass import unique_labels
from scipy.optimize import minimize

from .utils.som_hyparams import get_som_hyparams
from ._somd import SomDetector

# TODO: assert continuity
# TODO: som params

class ParallelAnomalousNudge(BaseEstimator):
    """
    Parallel Anomalous Nudge (PAN) for detecting novelties.
    """

    def __init__(self, scaler=StandardScaler(), normal_label=0, abnormal_label=1, nu=0.5, omega=2.0, random_seed=None, verbose=False):

        self.scaler = scaler
        self.normal_label = normal_label
        self.abnormal_label = abnormal_label
        self.nu = nu
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

        # Create ranking of abnormal training data

        X_abnormal = X_partitions[self.abnormal_label]
        self.X_abnormal_sample_n_ = len(X_abnormal)
        self.X_abnormal_deviations_ranked_ = sorted(abs(self._score_components(X_abnormal)[:, self.abnormal_label_idx_].ravel()))


        # ::: End of fitting :::


        # Obtain offset

        X_normal = X_partitions[self.normal_label]
        X_normal_scores = self.score_samples(X_normal)

        rho_initial = np.median(X_normal_scores)
        optim_res = minimize(self.__nu_loss, x0=[rho_initial], args=(X_normal_scores, self.nu), bounds=[(None, 0)])
        self.offset_ = optim_res.x[0]

        if self.verbose:
            print(f"\nOffset is calculated as:\t", self.offset_, "\n")

        return self
    
    def score_samples(self, X):
        """
        Opposite of the deviation of X measured from the closest reference point (best-matching unit, BMU) of the Normal SOM representation,
        boosted by a nudge factor based on deviation from the Abnormal representation.
        The nudge is determined by how samples rank in historical abnormal samples' deviation from the abnormal representation.
        The bigger is better, i.e. zero being the maximum value a sample can score, the closer the score is to zero, the more it is considered as an inlier.
        """
        check_is_fitted(self)
        X = self._validate_data(X)

        X_score = self._score_components(X)
        X_normal_score = X_score[:, self.normal_label_idx_].ravel()
        X_abnormal_score = X_score[:, self.abnormal_label_idx_].ravel()

        X_anomalous_rank = np.searchsorted(self.X_abnormal_deviations_ranked_, abs(X_abnormal_score)) + 1
        X_nudged_score = self.__internal_nudged_score_formula(X_normal_score, X_anomalous_rank)

        return np.array(X_nudged_score)
    
    def decision_function(self, X):
        return self.score_samples(X) - self.offset_
    
    def predict(self, X):
        """
        Perform classification on samples in X.
        A label of +1 or -1 is returned for inliers and outliers, respectively.
        """

        scores = self.decision_function(X)
        cnd_inlier = scores >= 0

        y_pred = np.zeros_like(scores, dtype=np.intp)
        y_pred[cnd_inlier] = 1
        y_pred[~cnd_inlier] = -1

        return y_pred

    def _score_components(self, X):
        """
        Opposite of the deviation of X measured from the closest reference point (best-matching unit, BMU) of the trained SOM representation, for all representation, in a tuple format.
        The bigger is better, i.e. zero being the maximum value a sample can score, the closer the score is to zero, the more it is considered as an inlier.
        """
        check_is_fitted(self)
        X = self._validate_data(X)

        X_dual_scaled = np.array([self.scalers_[c].transform(X) for c in self.classes_]).T
        X_dual_scores = np.array([est.score_samples(scaled_data) for est, scaled_data in zip(self.estimators_, X_dual_scaled.T)]).T

        return X_dual_scores

    def _score_component_normal(self, X):
        return self._score_components(X)[:, self.normal_label_idx_].ravel()
    
    def _score_component_abnormal(self, X):
        return self._score_components(X)[:, self.abnormal_label_idx_].ravel()

    def __sample_nudge_amount(self, X_rank):
        multiplier = ((self.X_abnormal_sample_n_ + 1) - X_rank) / self.X_abnormal_sample_n_
        return (multiplier * (self.omega - 1)) + 1

    def __internal_nudged_score_formula(self, X_normal_score, X_anomalous_rank):
        return X_normal_score * self.__sample_nudge_amount(X_anomalous_rank)

    def __nu_loss(self, rho, scores, nu):
        hinge_loss = np.maximum(0, rho - scores)
        boundary_penalty = nu * rho
        return np.mean(hinge_loss) - boundary_penalty
