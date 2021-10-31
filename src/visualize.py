import matplotlib.pyplot as plt  # importing the required modules
import numpy as np
from datagenerate import dataset
from plot import plot_adaboost
from boosting import AdaBoost


class VisualizeModel:
    """Visualizing Designed for model
     """

    def __init__(self, x, y):
        self.x = x["Data"]
        self.x_label = x["Label"]
        self.y = y["Data"]
        self.y_label = y["Label"]

    def scatter_hist(self, ax, ax_histx, ax_histy):
        """reference https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_hist.html#sphx-glr-gallery
        -lines-bars-and-markers-scatter-hist-py """

        # no labels
        ax_histx.tick_params(axis="x", labelbottom=self.x_label)
        ax_histy.tick_params(axis="y", labelleft=self.y_label)

        # the scatter plot:
        ax.scatter(self.x, self.y)

        # now determine nice limits by hand:
        range_x = np.subtract(self.x.max(), self.x.min())
        range_y = np.subtract(self.y.max(), self.y.min())
        bin_width_x = range_x / np.log(self.x.size)
        bin_width_y = range_y / np.log(self.y.size)
        bins_x = np.arange(self.x.min(), self.x.max(), bin_width_x)
        bins_y = np.arange(self.y.min(), self.y.max(), bin_width_y)
        ax_histx.hist(self.x, bins=bins_x)
        ax_histy.hist(self.y, bins=bins_y, orientation='horizontal')

    def plot_scatter_hist(self) -> object:
        # Definitions for the axes
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        spacing = 0.005

        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom + height + spacing, width, 0.2]
        rect_histy = [left + width + spacing, bottom, 0.2, height]

        # start with a square Figure
        fig = plt.figure(figsize=(8, 8))

        ax: plt = fig.add_axes(rect_scatter)
        ax_histx: plt = fig.add_axes(rect_histx, sharex=ax)
        ax_histy: plt = fig.add_axes(rect_histy, sharey=ax)

        self.scatter_hist(ax, ax_histx, ax_histy)
        # use the previous defined function
        plt.show()


class Scatter2Dims:

    @staticmethod
    def plot_clf(clf: AdaBoost):
        dx = clf.errors
        dy = clf.stump_weights
        x = {"Label": 'Stumps Error', "Data": dx}
        y = {"Label": 'Stumps Weight', "Data": dy}
        p2 = VisualizeModel(x, y)
        p2.plot_scatter_hist()

    @staticmethod
    def plot_2dcoordinate(X):
        dx, dy = zip(*X)
        x = {"Label": 'x axis', "Data": np.asarray(dx)}
        y = {"Label": 'y axis', "Data": np.asarray(dy)}
        p3 = VisualizeModel(x, y)
        p3.plot_scatter_hist()


def truncated_adaboost(clf: AdaBoost,
                       t: int):  # Truncate a fitted AdaBoost up to (and including) a particular iteration
    assert t > 0  # t must be a positive integer
    from copy import deepcopy
    new_clf = deepcopy(clf)
    new_clf.stumps = clf.stumps[:t]
    new_clf.stump_weights = clf.stump_weights[:t]
    return new_clf


def plot_iter_adaboost(train_data:np.ndarray, train_label:np.ndarray, clf:AdaBoost, iters):
    """
    Plot weak learner and cumulaive strong learner at each iteration.
    :type train_data: np.ndarray
    :type train_label: np.ndarray
    :type clf: AdaBoost
    :type iters: int
    :return:
    """

    fig, axes = plt.subplots(figsize=(8, iters * 3),  # larger grid
                             nrows=iters,
                             ncols=2,
                             sharex=True,
                             dpi=100)

    fig.set_facecolor('white')

    _ = fig.suptitle('Decision boundaries after every iteration')  # Plot the decision boundaries
    for i in range(iters):
        ax1, ax2 = axes[i]

        _ = ax1.set_title(f'Weak learner at iteration {i + 1}')  # Plot weak learner
        plot_adaboost(train_data, train_label, clf.stumps[i],
                      sample_weights=clf.sample_weights[i],
                      annotate=False, ax=ax1)

        trunc_clf = truncated_adaboost(clf, t=i + 1)  # Plot strong learner
        _ = ax2.set_title(f'Strong learner at iteration {i + 1}')
        plot_adaboost(train_data, train_label, trunc_clf,
                      sample_weights=clf.sample_weights[i],
                      annotate=False, ax=ax2)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()


X, y = dataset(100, 10, 2)
Scatter2Dims.plot_2dcoordinate(X)
clf = AdaBoost().fit(X, y, iters=10)
Scatter2Dims.plot_clf(clf)
plot_iter_adaboost(X, y, clf, 5)
