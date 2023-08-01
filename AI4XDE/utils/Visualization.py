import matplotlib.pyplot as plt


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


def set_axes_2D(axes, x_limit, y_limit, x_label, y_label, title):
    kwargs = [x_limit, y_limit, x_label, y_label, title]
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


def plot_data_2D(X, **kwargs):
    axes = set_axes_2D(**kwargs)
    axes.scatter(X[:, 0], X[:, 1])
    return axes


def plot_heatmap_2D(X, y, shape, **kwargs):
    axes = set_axes_2D(**kwargs)
    return axes.pcolormesh(
        X[:, 0].reshape(shape[0], shape[1]),
        X[:, 1].reshape(shape[0], shape[1]),
        y.reshape(shape[0], shape[1]),
        cmap="rainbow",
    )
