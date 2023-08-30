import numpy as np
import deepxde.backend as bkd
from deepxde.backend import torch


def transform_uniform_to_normal_2D(X):
    max_X = np.max(X, axis=0)
    min_X = np.min(X, axis=0)
    X = (X - min_X) / (max_X - min_X)

    X_new = np.zeros_like(X)
    X_new[:, 0] = (-2 * np.log(X[:, 0])) ** (1.0 / 2) * np.cos(2 * np.pi * X[:, 1])
    X_new[:, 1] = (-2 * np.log(X[:, 0])) ** (1.0 / 2) * np.sin(2 * np.pi * X[:, 1])
    return X_new


def transform_normal_to_truncated_normal_on_geomtime(X, geomtime, mul=0, sigma=1):
    X = X * sigma + mul
    X = X[geomtime.inside(X)]
    return X


def ntk(PDECase, solver, x=None, bc_point=False, compute="full"):
    # TODO: support other backends
    if bkd.backend_name != "pytorch":
        raise RuntimeError("NTK is only supported for PyTorch >= 2.0")

    if x is None:
        if bc_point:
            x = PDECase.geomtime.random_boundary_points(PDECase.NumDomain)
        else:
            x = PDECase.geomtime.random_points(PDECase.NumDomain)
        x = bkd.from_numpy(x)

    if type(x) is np.ndarray:
        x = bkd.from_numpy(x)

    # functorch
    # from functorch import make_functional, vmap, jacrev

    # fnet, params = make_functional(solver.model.net)

    # def fnet_single(params, x):
    #     y = fnet(params, x.unsqueeze(0)).squeeze(0)
    #     print(y.shape)
    #     print(y)
    #     return y

    # jac1 = vmap(jacrev(fnet_single), (None, 0))(params, x)
    # jac1 = [j.flatten(2) for j in jac1]

    # jac2 = vmap(jacrev(fnet_single), (None, 0))(params, x)
    # jac2 = [j.flatten(2) for j in jac2]

    # pytorch>=2.0
    from torch.func import functional_call, vmap, jacrev

    params = dict(solver.model.net.named_parameters())

    def fnet_single_torch(params, x):
        y = functional_call(solver.model.net, params, x.unsqueeze(0)).squeeze(0)
        return y

    jac1 = vmap(jacrev(fnet_single_torch), (None, 0))(params, x)
    jac1 = jac1.values()
    jac1 = [j.flatten(2) for j in jac1]

    jac2 = vmap(jacrev(fnet_single_torch), (None, 0))(params, x)
    jac2 = jac2.values()
    jac2 = [j.flatten(2) for j in jac2]

    # Compute J(x1) @ J(x2).T
    einsum_expr = None
    if compute == "full":
        einsum_expr = "Naf,Mbf->NMab"
    elif compute == "trace":
        einsum_expr = "Naf,Maf->NM"
    elif compute == "diagonal":
        einsum_expr = "Naf,Maf->NMa"
    else:
        assert False

    result = torch.stack(
        [torch.einsum(einsum_expr, j1, j2) for j1, j2 in zip(jac1, jac2)]
    )
    result = result.sum(0)
    return result.squeeze()


def eig_of_ntk(PDECase, solver, sort=True, **kwargs):
    ntk_result = ntk(PDECase, solver, **kwargs)
    lambda_K, eigvec_K = np.linalg.eig(bkd.to_numpy(ntk_result))
    if sort:
        lambda_K = np.sort(np.real(lambda_K))[::-1]
        eigvec_K = eigvec_K[:, np.argsort(np.real(lambda_K))[::-1]]
    return lambda_K, eigvec_K


def visualize_ntk(PDECase, solver, **kwargs):
    import matplotlib.pyplot as plt

    lambda_K, eigvec_K = eig_of_ntk(PDECase, solver, **kwargs)
    # Visualize the eigenvectors of the NTK
    fig, axs = plt.subplots(2, 3, figsize=(12, 6))
    X = np.linspace(0, 1, lambda_K.shape[0])
    axs[0, 0].plot(X, np.real(eigvec_K[:, 0]))
    axs[0, 1].plot(X, np.real(eigvec_K[:, 1]))
    axs[0, 2].plot(X, np.real(eigvec_K[:, 2]))
    axs[1, 0].plot(X, np.real(eigvec_K[:, 3]))
    axs[1, 1].plot(X, np.real(eigvec_K[:, 4]))
    axs[1, 2].plot(X, np.real(eigvec_K[:, 5]))
    plt.show()

    # Visualize the eigenvalues of the NTK
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(lambda_K.squeeze())
    plt.xscale("log")
    plt.yscale("log")
    ax.set_xlabel("index")
    ax.set_ylabel(r"$\lambda$")
    plt.show()
