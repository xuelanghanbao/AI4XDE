import matplotlib.pyplot as plt


class Visualization_1D:
    def plot_1D_result(PDECase, solver, axes=None, exact=True, xlabel="x", ylabel="y"):
        X, y = PDECase.get_testdata()
        if axes is None:
            fig, axes = plt.subplots()
        if exact:
            axes.plot(X, y, label="Exact")
        axes.plot(X, solver.model.predict(X), "--", label="Prediction")
        axes.legend()
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.set_title(PDECase.name)
        return fig, axes


class Visualization_2D:
    def __init__(self, x_limit=None, y_limit=None, x_label=None, y_label=None):
        self.x_limit = x_limit
        self.y_limit = y_limit
        self.x_label = x_label
        self.y_label = y_label

    def set_axes_2D(self, axes=None, title=None):
        kwargs = [self.x_limit, self.y_limit, self.x_label, self.y_label, title]
        if axes is None:
            fig, axes = plt.subplots()
        fun = [
            axes.set_xlim,
            axes.set_ylim,
            axes.set_xlabel,
            axes.set_ylabel,
            axes.set_title,
        ]
        for i, f in enumerate(fun):
            if kwargs[i] is not None:
                f(kwargs[i])
        return axes

    def plot_data_2D(self, X, **kwargs):
        axes = self.set_axes_2D(**kwargs)
        axes.scatter(X[:, 0], X[:, 1])
        return axes

    def plot_heatmap_2D(self, X, y, shape, **kwargs):
        axes = self.set_axes_2D(**kwargs)
        return axes.pcolormesh(
            X[:, 0].reshape(shape[0], shape[1]),
            X[:, 1].reshape(shape[0], shape[1]),
            y.reshape(shape[0], shape[1]),
            cmap="rainbow",
        )
