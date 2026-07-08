import numpy as np

from sklearn.base import BaseEstimator
from .utils.som import train_som

class SomDetector(BaseEstimator):
    """
    Self-Organizing Map (SOM)-based estimator for measuring deviation from a single class representation.
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

    def fit(self, X, y=None):
        X = self._validate_data(X)
        self.X_ = X

        som_hyperparams = {
            "d1": self.d1,
            "d2": self.d2,
            "sigma": self.sigma,
            "topology": self.topology,
            "learning_rate": self.learning_rate,
            "num_iteration": self.num_iteration,
            "decay_function": self.decay_function,
            "sigma_decay_function": self.sigma_decay_function,
            "use_epochs": self.use_epochs,
            "random_order": self.random_order,
            "random_seed": self.random_seed,
            "verbose": self.verbose,
        }

        som = train_som(X, **som_hyperparams)
        self.som_ = som

        return self

    def score_samples(self, X):
        """
        Opposite of the deviation of X measured from the closest reference point (best-matching unit, BMU) of the trained representation.
        The bigger is better, i.e. zero being the maximum value a sample can score, the closer the score is to zero, the more it is considered as an inlier.
        """
        X = self._validate_data(X)

        quantization_errors = np.linalg.norm(X - self.som_.quantization(X), axis=1)
        return (quantization_errors * -1)
