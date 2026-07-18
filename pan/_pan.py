import numpy as np

from sklearn.base import BaseEstimator, OutlierMixin, ClassifierMixin, check_is_fitted, clone
from sklearn.preprocessing import StandardScaler
from sklearn.utils.multiclass import unique_labels
from scipy.optimize import minimize

from ._somd import SomRepresentation

# TODO: assert continuity

class ParallelAnomalousNudge(OutlierMixin, BaseEstimator):
    """
    Parallel Anomalous Nudge (PAN) for detecting novelties.
    """

    def __init__(self, estimators=None, scaler=StandardScaler(), nu=0.5, omega=2.0, random_seed=None, verbose=False):

        self.estimators = estimators
        self.scaler = scaler
        self.nu = nu
        self.omega = omega
        self.random_seed = random_seed
        self.verbose = verbose

    def fit(self, X, y, continuity_violation_limit=0.1):
        X, y = self._validate_data(X, y)

        self.X_ = X
        self.y_ = y
        self.classes_ = unique_labels(y).astype(int)

        if self.estimators is not None:
            assert (len(self.classes_) == 2) and (len(self.estimators) == 2), "PAN currently supports two classes."
        
        self.normal_label_ = 0
        self.abnormal_label_ = 1
        self.normal_label_idx_ = np.argwhere(self.classes_ == self.normal_label_)
        self.abnormal_label_idx_ = np.argwhere(self.classes_ == self.abnormal_label_)

        self.scalers_ = {}
        self.estimators_ = {}
        self.X_partitions_ = {}
        self.X_partitions_scaled_ = {}

        # Fit partition-wise scalers, transform X, learn SOM-based detectors
        for c in self.classes_:
            XP = X[y == c]
            scaler = clone(self.scaler).fit(XP)
            XP_scaled = scaler.transform(XP)

            # Use the parameters of the given estimator blueprint or let it derive internally
            estimator = self.estimators[c] if self.estimators is not None else SomRepresentation(nu=self.nu, random_seed=self.random_seed, verbose=self.verbose)
            estimator.fit(XP_scaled)

            self.X_partitions_[c] = XP
            self.X_partitions_scaled_[c] = XP_scaled
            self.scalers_[c] = scaler
            self.estimators_[c] = estimator

        # Create ranking of abnormal training data

        X_abnormal = self.X_partitions_[self.abnormal_label_]
        self.X_abnormal_sample_n_ = len(X_abnormal)
        self.X_abnormal_deviations_ranked_ = sorted(abs(self._score_components(X_abnormal)[:, self.abnormal_label_idx_].ravel()))


        # ::: End of fitting :::


        # Obtain offset

        X_normal = self.X_partitions_[self.normal_label_]
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

        X_nudged_score = self.__nudge_normal_component(X_normal_score, X_abnormal_score)

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

    def fit_predict(self, X, y=None, **kwargs):
        return self.fit(X, y, **kwargs).predict(X)

    def wrapAsClassifier(self):
        return ParallelAnomalousNudgeClassifierWrapper(self)

    def _score_components(self, X):
        """
        Opposite of the deviation of X measured from the closest reference point (best-matching unit, BMU) of the trained SOM representation, for all representation, in a tuple format.
        The bigger is better, i.e. zero being the maximum value a sample can score, the closer the score is to zero, the more it is considered as an inlier.
        """
        check_is_fitted(self)
        X = self._validate_data(X)

        X_dual_scores = np.empty((len(X), 2))

        for c_idx, c in enumerate(self.classes_):
            XC_scaled = self.scalers_[c].transform(X)
            XC_scores = self.estimators_[c].score_samples(XC_scaled)
            X_dual_scores[:, c_idx] = XC_scores

        return X_dual_scores

    def _score_component_normal(self, X):
        return self._score_components(X)[:, self.normal_label_idx_].ravel()
    
    def _score_component_abnormal(self, X):
        return self._score_components(X)[:, self.abnormal_label_idx_].ravel()

    def __nudge_normal_component(self, X_normal_score, X_abnormal_score):
        X_anomalous_rank = np.searchsorted(self.X_abnormal_deviations_ranked_, abs(X_abnormal_score)) + 1
        X_nudged_score = self.__internal_nudge_formula(X_normal_score, X_anomalous_rank)
        return X_nudged_score

    def __internal_nudge_formula(self, X_normal_score, X_anomalous_rank):
        return X_normal_score * self.__nudge_factor(X_anomalous_rank)

    def __nudge_factor(self, X_rank):
        multiplier = ((self.X_abnormal_sample_n_ + 1) - X_rank) / self.X_abnormal_sample_n_
        return (multiplier * (self.omega - 1)) + 1

    def __nu_loss(self, rho, scores, nu):
        hinge_loss = np.maximum(0, rho - scores)
        boundary_penalty = nu * rho
        return np.mean(hinge_loss) - boundary_penalty

    def _more_tags(self):
        return {"requires_y": True}

class ParallelAnomalousNudgeClassifierWrapper(ClassifierMixin, BaseEstimator):

    def __init__(self, estimator:ParallelAnomalousNudge):
        self.estimator = estimator
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        self.estimator.fit(X, y)
        return self

    def decision_function(self, X):
        return -self.estimator.decision_function(X)

    def predict(self, X):
        return np.where(self.estimator.predict(X) == -1, 1, 0)
