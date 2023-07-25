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
