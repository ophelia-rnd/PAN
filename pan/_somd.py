import numpy as np

from sklearn.base import BaseEstimator, check_is_fitted, clone
from minisom import MiniSom

class SomDetector(BaseEstimator):
    """
    Self-Organizing Map (SOM)-based estimator for measuring deviation from a single class representation.
    """

    def __init__(self, som:MiniSom=None, d1=4, d2=4, sigma=1.0, topology="rectangular", learning_rate=0.5, num_iteration=20,
                    decay_function="linear_decay_to_zero", sigma_decay_function="asymptotic_decay",
                    use_epochs=True, random_order=True, random_seed=None, verbose=False):
        self.som = som
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

        if self.som is not None:
            self.som_ = clone(self.som)

        else:
            som_hyperparams = {
                "x": self.d1,
                "y": self.d2,
                "input_len": X.shape[1],
                "sigma": self.sigma,
                "topology": self.topology,
                "learning_rate": self.learning_rate,
                "decay_function": self.decay_function,
                "sigma_decay_function": self.sigma_decay_function,
                "random_seed": self.random_seed
            }

            som_fit_hyperparams ={
                "num_iteration": self.num_iteration,
                "random_order": self.random_order,
                "use_epochs": self.use_epochs,
                "verbose": self.verbose
            }

            som = self.__train_som(X, som_hyperparams, som_fit_hyperparams)

            if self.verbose:
                QE, TE, QE_ROUNDED, TE_ROUNDED = self.__som_quality(som, X)
                print("\nQuality of SOM:")
                print(f"Quantization error:\t{QE}")
                print(f"Topographic error:\t{TE}")
                print(f"Quantization error (rounded):\t{QE_ROUNDED}")
                print(f"Topographic error (rounded):\t{TE_ROUNDED}")

            self.som_ = som

        return self

    def score_samples(self, X):
        """
        Opposite of the deviation of X measured from the closest reference point (best-matching unit, BMU) of the trained representation.
        The bigger is better, i.e. zero being the maximum value a sample can score, the closer the score is to zero, the more it is considered as an inlier.
        """
        check_is_fitted(self)
        X = self._validate_data(X)

        quantization_errors = np.linalg.norm(X - self.som_.quantization(X), axis=1)
        return (quantization_errors * -1)

    def __train_som(self, X, hyperparams, fit_hyperparams):
        som = MiniSom(**hyperparams)
        som.random_weights_init(X)
        som.train(X, **fit_hyperparams)
        return som

    def __som_quality(self, som:MiniSom, X, digits=3):
        QE = som.quantization_error(X)
        TE = som.topographic_error(X)
        return QE, TE, np.round(QE, digits), np.round(TE, digits)
