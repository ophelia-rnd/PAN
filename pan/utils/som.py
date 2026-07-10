import numpy as np
from minisom import MiniSom

def train_som(X, d1, d2, sigma, topology, learning_rate, num_iteration,
			  decay_function, sigma_decay_function, use_epochs, random_order,
			  random_seed, verbose=False):

    som = MiniSom(d1, d2, input_len=X.shape[1], sigma=sigma, topology=topology, learning_rate=learning_rate,
                    decay_function=decay_function, sigma_decay_function=sigma_decay_function, random_seed=random_seed)
    som.random_weights_init(X)
    som.train(X, num_iteration=num_iteration, random_order=random_order, use_epochs=use_epochs, verbose=verbose)

    if verbose:
        QE, TE, QE_ROUNDED, TE_ROUNDED = calc_som_quality(som, X)
        print("\nQuality of SOM:")
        print(f"Quantization error:\t{QE}")
        print(f"Topographic error:\t{TE}")
        print(f"Quantization error (rounded):\t{QE_ROUNDED}")
        print(f"Topographic error (rounded):\t{TE_ROUNDED}")

    return som

def calc_som_quality(som:MiniSom, X, digits=3):
	QE = som.quantization_error(X)
	TE = som.topographic_error(X)
	return QE, TE, np.round(QE, digits), np.round(TE, digits)
