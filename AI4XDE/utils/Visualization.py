import matplotlib.pyplot as plt


class Visualization_1D:
    def __init__(self, x_label=None, y_label=None):
        self.x_label = x_label
        self.y_label = y_label

    def set_axes_1D(self, axes=None, title=None):
        kwargs = [self.x_label, self.y_label, title]
        if axes is None:
            fig, axes = plt.subplots()
        fun = [axes.set_xlabel, axes.set_ylabel, axes.set_title]
        for i, f in enumerate(fun):
            if kwargs[i] is not None:
                f(kwargs[i])
        return axes

    def plot_line_1D(self, PDECase, solver, exact=True, **kwargs):
        axes = self.set_axes_1D(**kwargs, title=PDECase.name)

        X, y = PDECase.get_testdata()
        if exact:
            axes.plot(X, y, label="Exact")
        axes.plot(X, solver.model.predict(X), "--", label="Prediction")
        axes.legend()
        return axes


class Visualization_2D:
    def __init__(
        self,
        x_limit=None,
        y_limit=None,
        x_label=None,
        y_label=None,
        feature_transform=None,
    ):
        self.x_limit = x_limit
        self.y_limit = y_limit
        self.x_label = x_label
        self.y_label = y_label
        self.feature_transform = feature_transform

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
        if self.feature_transform is not None:
            X = self.feature_transform(X)
        axes.scatter(X[:, 0], X[:, 1])
        return axes

    def plot_heatmap_2D(self, X, y, shape, **kwargs):
        axes = self.set_axes_2D(**kwargs)
        if self.feature_transform is not None:
            X = self.feature_transform(X)
        return axes.pcolormesh(
            X[:, 0].reshape(shape[0], shape[1]),
            X[:, 1].reshape(shape[0], shape[1]),
            y.reshape(shape[0], shape[1]),
            cmap="rainbow",
        )
