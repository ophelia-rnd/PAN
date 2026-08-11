import numpy as np

from minisom import MiniSom
from sklearn.base import BaseEstimator, check_is_fitted
from scipy.optimize import minimize
from minisom_representation import SomRepresentation


class SomRepresentationEstimator(BaseEstimator):
    """
    Self-Organizing Map (SOM)-based estimator for learning a representation and for measuring deviation from it.
    """

    def __init__(self, som_representation:SomRepresentation, nu=0.5, random_seed=None, verbose=False):
        self.nu = nu
        self.som_representation = som_representation
        self.random_seed = random_seed
        self.verbose = verbose

    @classmethod
    def from_som_representation(cls, som_representation:SomRepresentation, nu=0.5, random_seed=None, verbose=False):
        return cls(som_representation=som_representation, nu=nu, random_seed=random_seed, verbose=verbose)

    @classmethod
    def with_derived_som_representation(cls, X, nu=0.5, random_seed=None, verbose=False):
        som_representation = SomRepresentation.with_derived_params(X, random_seed=random_seed, verbose=verbose)
        return cls(som_representation=som_representation, nu=nu, random_seed=random_seed, verbose=verbose)

    @property
    def som(self) -> MiniSom:
        assert self.som_representation is not None, "SomRepresentationEstimator must be set."
        assert self.som_representation.som is not None, "SomRepresentationEstimator must be trained."
        return self.som_representation.som

    def fit(self, X, y=None):
        X = self._validate_data(X)
        self.som_representation.fit(X)

        X_scores = self.score_samples(X)
        rho_initial = np.median(X_scores)
        optim_res = minimize(self.__nu_loss, x0=[rho_initial], args=(X_scores, self.nu), bounds=[(None, 0)])
        self.offset_ = optim_res.x[0]

        if self.verbose:
            print("\n", "An SOM representation estimator has been fitted as follows:")
            print("-----------------------------------------------------------------", "\n")
            print("Hyperparameters:", "\n")
            print(self.get_params(), "\n")

            print("Learned parameters:", "\n")
            print(f"Offset:\t{self.offset_}")

        return self

    def score_samples(self, X):
        """
        Opposite of the deviation of X measured from the closest reference point (best-matching unit, BMU) of the trained representation.
        The bigger is better, i.e. zero being the maximum value a sample can score, the closer the score is to zero, the more it is considered as an inlier.
        """
        check_is_fitted(self)
        X = self._validate_data(X)

        quantization_errors = np.linalg.norm(X - self.som.quantization(X), axis=1)
        return (quantization_errors * -1)

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

    def __nu_loss(self, rho, scores, nu):
        hinge_loss = np.maximum(0, rho - scores)
        boundary_penalty = nu * rho
        return np.mean(hinge_loss) - boundary_penalty
