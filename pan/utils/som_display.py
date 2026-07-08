from minisom import MiniSom

class SomLatticeDisplay:

    def __init__(self):
        self.node_weights_ = None

    def plot(self, node_weights=None, ax=None, xlabel=None, ylabel=None, **kwargs):
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        nweights = node_weights if node_weights is not None else self.node_weights_
        assert nweights is not None, "Must provide `node_weights`."

        if ax is None:
            _, ax = plt.subplots()

        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        ax.scatter(*nweights.T, **kwargs)

        rows, cols = nweights.shape[:2]
        feat_x, feat_y = 0, 1

        kwargs.pop('marker')

        for i in range(rows):
            for j in range(cols):
                current = nweights[i, j]

                if i + 1 < rows:
                    neighbor = nweights[i+1, j]
                    ax.plot([current[feat_x], neighbor[feat_x]],
                            [current[feat_y], neighbor[feat_y]],
                            linestyle='-', **kwargs)

                if j + 1 < cols:
                    neighbor = nweights[i, j+1]
                    ax.plot([current[feat_x], neighbor[feat_x]],
                            [current[feat_y], neighbor[feat_y]],
                            linestyle='-', **kwargs)

        self.ax_ = ax
        self.figure_ = ax.figure
        return self
    
    def from_som(self, som:MiniSom):
        node_weights = som.get_weights()
        self.node_weights_ = node_weights
        return self
    
    def from_values(self, node_weights):
        self.node_weights_ = node_weights
        return self
