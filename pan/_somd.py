import numpy as np

from sklearn.base import BaseEstimator, check_is_fitted, clone
from minisom import MiniSom
from scipy.optimize import minimize

from .utils.som_hyparams import calc_som_hyparams

# FIXME: name SomRepresentationAD ?
class SomRepresentation(BaseEstimator):
    """
    Self-Organizing Map (SOM)-based estimator for learning a representation and for measuring deviation from it.
    """

    def __init__(self, nu=0.5, d1=None, d2=None, sigma=None, topology="rectangular", learning_rate=0.5, num_iteration=20,
                    decay_function="asymptotic_decay", sigma_decay_function="asymptotic_decay",
                    neighborhood_function='gaussian', activation_distance='euclidean',
                    use_epochs=True, random_order=True, random_seed=None, verbose=False):
        self.nu = nu
        self.d1 = d1
        self.d2 = d2
        self.sigma = sigma
        self.topology = topology
        self.learning_rate = learning_rate
        self.num_iteration = num_iteration
        self.decay_function = decay_function
        self.sigma_decay_function = sigma_decay_function
        self.neighborhood_function = neighborhood_function
        self.activation_distance = activation_distance
        self.use_epochs = use_epochs
        self.random_order = random_order
        self.random_seed = random_seed
        self.verbose = verbose

    def fit(self, X, y=None):
        X = self._validate_data(X)
        self.X_ = X

        if not self.__main_params_set():
            hyparams = self.__derive_main_params(X)

        som_hyperparams = {
            "input_len": X.shape[1],
            "x": self.d1 if self.d1 is not None else hyparams.get("d1"),
            "y": self.d2 if self.d2 is not None else hyparams.get("d2"),
            "sigma": self.sigma if self.sigma is not None else hyparams.get("sigma"),
            "topology": self.topology,
            "learning_rate": self.learning_rate,
            "decay_function": self.decay_function,
            "sigma_decay_function": self.sigma_decay_function,
            "neighborhood_function": self.neighborhood_function,
            "activation_distance": self.activation_distance,
            "random_seed": self.random_seed
        }

        som_fit_hyperparams = {
            "num_iteration": self.num_iteration,
            "random_order": self.random_order,
            "use_epochs": self.use_epochs,
            "verbose": self.verbose
        }

        som = self.__train_som(X, som_hyperparams, som_fit_hyperparams)

        if self.verbose:
            print("\nFinal hyperparameters of SOM:")
            print({
                **som_hyperparams,
                **som_fit_hyperparams
            })

            QE, TE, QE_ROUNDED, TE_ROUNDED = self.__som_quality(som, X)
            print("\nQuality of SOM:")
            print(f"Quantization error:\t{QE}")
            print(f"Topographic error:\t{TE}")
            print(f"Quantization error (rounded):\t{QE_ROUNDED}")
            print(f"Topographic error (rounded):\t{TE_ROUNDED}")

        self.som_ = som

        X_scores = self.score_samples(X)
        rho_initial = np.median(X_scores)
        optim_res = minimize(self.__nu_loss, x0=[rho_initial], args=(X_scores, self.nu), bounds=[(None, 0)])
        self.offset_ = optim_res.x[0]

        if self.verbose:
            print(f"\nOffset is calculated as:\t", self.offset_, "\n")

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

    def __main_params_set(self):
        return not (self.d1 is None or self.d2 is None or self.sigma is None)

    def __derive_main_params(self, X):
        hyparams = calc_som_hyparams(X, verbose=self.verbose)
        return hyparams

    def __train_som(self, X, hyperparams, fit_hyperparams):
        som = MiniSom(**hyperparams)
        som.random_weights_init(X)
        som.train(X, **fit_hyperparams)
        return som

    def __som_quality(self, som:MiniSom, X, digits=3):
        QE = som.quantization_error(X)
        TE = som.topographic_error(X)
        return QE, TE, np.round(QE, digits), np.round(TE, digits)

    def __nu_loss(self, rho, scores, nu):
        hinge_loss = np.maximum(0, rho - scores)
        boundary_penalty = nu * rho
        return np.mean(hinge_loss) - boundary_penalty
