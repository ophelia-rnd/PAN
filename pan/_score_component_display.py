import numpy as np

from typing import Literal
from sklearn.inspection import DecisionBoundaryDisplay
from pan import ParallelAnomalousNudge

style_train = {
    "marker": "o",
    "s": 500,
    "edgecolor": "black",
    "color": "none",
    "linewidth": 2,
    "alpha": .3,
    "zorder": 11,
    "label": "train",
}

style_train_anomaly = {
    **style_train,
    "edgecolor": "crimson",
    "label": "train (anomaly)",
}

style_val = {
    "marker": "s",
    "s": 300,
    "edgecolor": "black",
    "color": "none",
    "linewidth": 2,
    "alpha": .4,
    "zorder": 12,
    "label": "validation",
}

style_val_anomaly = {
    **style_val,
    "edgecolor": "crimson",
    "label": "validation (anomaly)",
}

style_test = {
    "marker": "^",
    "s": 300,
    "edgecolor": "royalblue",
    "color": "none",
    "linewidth": 2,
    "alpha": .8,
    "zorder": 13,
    "label": "test",
}

style_test_anomaly = {
    **style_test,
    "edgecolor": "crimson",
    "label": "test (anomaly)",
}

class ScoreComponentDisplay():

    @classmethod
    def from_estimator(cls, estimator:ParallelAnomalousNudge, X, threshold=None,
                       xlabel="Deviation from Normal", ylabel="Deviation from Anomalous",
                       ax=None, threshold_style={}, **kwargs):
        from matplotlib.colors import ListedColormap

        X_score_comp_1 = estimator._score_component_normal(X)
        X_score_comp_2 = estimator._score_component_abnormal(X)

        x0_min, x0_max = X_score_comp_1.min() - 0.5, 0
        x1_min, x1_max = X_score_comp_2.min() - 0.5, 0
        xx0, xx1 = np.meshgrid(np.linspace(x0_min, x0_max, 300), np.linspace(x1_min, x1_max, 300))

        XX_scores = estimator._ParallelAnomalousNudge__nudge_normal_component(xx0.ravel(), xx1.ravel()) \
            .reshape(xx0.shape)

        db_display = DecisionBoundaryDisplay(
            xx0=xx0 * -1,
            xx1=xx1 * -1,
            response=XX_scores,
            xlabel=xlabel,
            ylabel=ylabel,
        )

        threshold = threshold if threshold is not None else estimator.offset_
        kwargs.setdefault("cmap", ListedColormap(["#b87bc9", "#6cd4d9"]))

        db_display.plot(
            ax=ax,
            plot_method="contourf",
            levels=[XX_scores.min(), threshold, XX_scores.max()],
            zorder=0,
            **kwargs,
        )

        display = cls()
        display.estimator_ = estimator
        display.ax_ = db_display.ax_
        display.figure_ = display.ax_.figure

        display.ax_.contour(
            db_display.xx0,
            db_display.xx1,
            db_display.response,
            levels=[threshold],
            zorder=100,
            **threshold_style,
        )

        return display

    def plot_samples(self, X, style_preset:Literal["train", "train_anomaly", "val", "val_anomaly", "test", "test_anomaly"]=None, **kwargs):
        assert hasattr(self, "estimator_") and self.estimator_ is not None
        assert hasattr(self, "ax_") and self.ax_ is not None

        _kwargs = {}
        if style_preset is not None:
            if style_preset == "train": _kwargs = {**style_train}
            elif style_preset == "train_anomaly": _kwargs = {**style_train_anomaly}
            elif style_preset == "val": _kwargs = {**style_val}
            elif style_preset == "val_anomaly": _kwargs = {**style_val_anomaly}
            elif style_preset == "test": _kwargs = {**style_test}
            elif style_preset == "test_anomaly": _kwargs = {**style_test_anomaly}
        _kwargs = {**_kwargs, **kwargs}

        X_score_comp_1 = self.estimator_._score_component_normal(X) * -1
        X_score_comp_2 = self.estimator_._score_component_abnormal(X) * -1

        ax = self.ax_
        ax.scatter(X_score_comp_1, X_score_comp_2, **_kwargs)
        ax.set_aspect("equal")

        return self
