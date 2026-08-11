import numpy as np

from sklearn.base import BaseEstimator, OutlierMixin, ClassifierMixin, check_is_fitted, clone
from sklearn.preprocessing import StandardScaler
from sklearn.utils.multiclass import unique_labels
from scipy.optimize import minimize

from ._som_estimator import SomRepresentationEstimator

# TODO: assert continuity

class ParallelAnomalousNudge(OutlierMixin, BaseEstimator):
    """
    Parallel Anomalous Nudge (PAN) for detecting novelties.
    """

    def __init__(self, estimators, scaler, nu=0.5, omega=2.0, normal_label=0, abnormal_label=1,
                 random_seed=None, verbose=False):

        self.estimators = estimators
        self.scaler = scaler
        self.nu = nu
        self.omega = omega
        self.normal_label = normal_label
        self.abnormal_label = abnormal_label
        self.random_seed = random_seed
        self.verbose = verbose

    @classmethod
    def from_estimators(cls, estimators:dict, scaler=StandardScaler(), nu=0.5, omega=2.0, normal_label=0, abnormal_label=1, random_seed=None, verbose=False):
        return cls(estimators=estimators, scaler=scaler, nu=nu, omega=omega, normal_label=normal_label, abnormal_label=abnormal_label, random_seed=random_seed, verbose=verbose)

    @classmethod
    def with_derived_estimators(cls, X, y, scaler=StandardScaler(), nu=0.5, omega=2.0, normal_label=0, abnormal_label=1, random_seed=None, verbose=False):
        unique_classes = unique_labels(y).astype(int)
        X_partitions = cls.__partition_data(X, y, unique_classes)
        X_partitions_scaled, _ = cls.__scale_partitions(X_partitions, unique_classes, scaler)
        estimators = {}

        for c in unique_classes:
            XP_scaled = X_partitions_scaled[c]
            estimator = SomRepresentationEstimator.with_derived_som_representation(X=XP_scaled, nu=nu, random_seed=random_seed, verbose=verbose)
            estimator.fit(XP_scaled)
            estimators[c] = estimator

        return cls(estimators=estimators, scaler=scaler, nu=nu, omega=omega, normal_label=normal_label, abnormal_label=abnormal_label, random_seed=random_seed, verbose=verbose)

    @classmethod
    def __partition_data(cls, X, y, unique_classes):
        X_partitions = {}
        for c in unique_classes:
            XP = X[y == c]
            X_partitions[c] = XP
        return X_partitions

    @classmethod
    def __scale_partitions(cls, X_partitions, unique_classes, scaler):
        X_partitions_scaled = {}
        scalers = {}
        for c, XP in [(c, X_partitions[c]) for c in unique_classes]:
            scaler = clone(scaler).fit(XP)
            XP_scaled = scaler.transform(XP)
            X_partitions_scaled[c] = XP_scaled
            scalers[c] = scaler
        return X_partitions_scaled, scalers

    def fit(self, X, y):
        X, y = self._validate_data(X, y)
        self.classes_ = unique_labels(y).astype(int)
        assert (len(self.classes_) == 2) and (len(self.estimators) == 2), "PAN currently supports two classes."
        
        self.normal_label_idx_ = np.argwhere(self.classes_ == self.normal_label)
        self.abnormal_label_idx_ = np.argwhere(self.classes_ == self.abnormal_label)

        X_partitions = self.__partition_data(X, y, self.classes_)
        X_partitions_scaled, scalers = self.__scale_partitions(X_partitions, self.classes_, self.scaler)
        estimators = {}

        # Fit partition-wise scalers, transform X, learn SOM-based detectors
        for c in self.classes_:
            XP_scaled = X_partitions_scaled[c]
            estimator = (
                clone(self.estimators[c])
                if self.estimators.get(c, None) is not None
                else SomRepresentationEstimator.with_derived_som_representation(X=XP_scaled, nu=self.nu, random_seed=self.random_seed, verbose=self.verbose)
            )
            estimator.fit(XP_scaled)
            estimators[c] = estimator

        self.scalers_ = scalers
        self.estimators_ = estimators

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
            print("\n", "A PAN estimator has been fitted as follows:")
            print("-----------------------------------------------------------------", "\n")
            print("Hyperparameters:", "\n")
            print(self.get_params(), "\n")

            print("Learned parameters:", "\n")
            print(f"Offset:\t{self.offset_}")

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
        return ParallelAnomalousNudgeClassifier(self)

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

class ParallelAnomalousNudgeClassifier(ClassifierMixin, BaseEstimator):

    NORMAL = 0
    ABNORMAL = 1

    def __init__(self, detector:ParallelAnomalousNudge):
        self.detector = detector

    def fit(self, X, y):
        self.detector.fit(X, y)
        self.classes_ = np.array([self.NORMAL, self.ABNORMAL])
        return self

    def decision_function(self, X):
        return self.detector.decision_function(X) * -1

    def predict(self, X):
        return np.where(self.decision_function(X) > 0, self.ABNORMAL, self.NORMAL)

    def unwrapNoveltyDetector(self):
        return self.detector
