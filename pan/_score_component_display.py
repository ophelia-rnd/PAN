import numpy as np

from sklearn.inspection import DecisionBoundaryDisplay
from pan import ParallelAnomalousNudge

class ScoreComponentDisplay():

    @classmethod
    def from_estimator(cls, estimator:ParallelAnomalousNudge, X,
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

        threshold = estimator.offset_
        kwargs.setdefault("cmap", ListedColormap(["crimson", "cornflowerblue"]))

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
            zorder=1,
            **threshold_style,
        )

        return display

    def plot_samples(self, X, **kwargs):
        assert hasattr(self, "estimator_") and self.estimator_ is not None
        assert hasattr(self, "ax_") and self.ax_ is not None

        X_score_comp_1 = self.estimator_._score_component_normal(X) * -1
        X_score_comp_2 = self.estimator_._score_component_abnormal(X) * -1

        ax = self.ax_
        ax.scatter(X_score_comp_1, X_score_comp_2, **kwargs)
        ax.set_aspect("equal")

        return self

    def plot_samples_by_class(self, X_train, X_test, y_train, y_test, normal_label=0, abnormal_label=1, **kwargs):
        return self \
            .plot_samples(X_train[y_train == normal_label],
                          marker="o", s=500, edgecolor="black", linewidth=2, color="none", alpha=.3, zorder=1, label="train (normal)") \
            .plot_samples(X_train[y_train == abnormal_label],
                          marker="X", s=500, edgecolor="lightcoral", linewidth=2, color="none", alpha=.5, zorder=2, label="train (anomaly)") \
            .plot_samples(X_test[y_test == normal_label],
                          marker="^", s=100, edgecolor="cornflowerblue", linewidth=2, color="none", alpha=.8, zorder=3, label="test (normal)") \
            .plot_samples(X_test[y_test == abnormal_label],
                          marker="^", s=100, edgecolor="crimson", linewidth=2, color="none", alpha=.8, zorder=4, label="test (anomaly)")
