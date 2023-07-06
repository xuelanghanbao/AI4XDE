"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import inspect
import numpy as np
import deepxde as dde
from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence, plot_objective
from ..solver.PDESolver import PINNSolver

class HPO(PINNSolver):
    def __init__(self, 
                 PDECase,
                 iterations=1000,
                 n_calls=50,
                 default_parameters=[1e-3, 4, 50, "sin"]):
        self.iterations = iterations
        self.n_calls = n_calls
        self.default_parameters = default_parameters
        self.ITERATION = 0
        self.d = self.get_layer_size(PDECase)
        super().__init__(name='HPO', PDECase=PDECase)
    
    def get_layer_size(self, PDECase):
        argspec = inspect.getfullargspec(PDECase.__init__)
        init_defaults = dict(zip(argspec.args[-len(argspec.defaults):], argspec.defaults))
        b = init_defaults['layer_size'][0]
        return b

    def create_model(self, config):
        learning_rate, num_dense_layers, num_dense_nodes, activation = config
        self.PDECase.net = self.PDECase.gen_net(
            [self.d] + [num_dense_nodes] * num_dense_layers + [1],
            activation,
            "Glorot uniform",
        )
        self.model = dde.Model(self.PDECase.data, self.PDECase.net)
        self.model.compile("adam", lr=learning_rate)
        return self.model
    
    def train_step(self):
        losshistory, train_state = self.model.train(iterations=self.iterations)
        train = np.array(losshistory.loss_train).sum(axis=1).ravel()
        test = np.array(losshistory.loss_test).sum(axis=1).ravel()
        metric = np.array(losshistory.metrics_test).sum(axis=1).ravel()

        error = test.min()
        return error

    def closure(self):
        dim_learning_rate = Real(low=1e-4, high=5e-2, name="learning_rate", prior="log-uniform")
        dim_num_dense_layers = Integer(low=1, high=10, name="num_dense_layers")
        dim_num_dense_nodes = Integer(low=5, high=500, name="num_dense_nodes")
        dim_activation = Categorical(categories=["sin", "sigmoid", "tanh"], name="activation")
        dimensions = [
            dim_learning_rate,
            dim_num_dense_layers,
            dim_num_dense_nodes,
            dim_activation,
        ]
        @use_named_args(dimensions=dimensions)
        def fitness(learning_rate, num_dense_layers, num_dense_nodes, activation):
            config = [learning_rate, num_dense_layers, num_dense_nodes, activation]

            print(self.ITERATION, "it number")
            # Print the hyper-parameters.
            print("learning rate: {0:.1e}".format(learning_rate))
            print("num_dense_layers:", num_dense_layers)
            print("num_dense_nodes:", num_dense_nodes)
            print("activation:", activation)
            print()

            # Create the neural network with these hyper-parameters.
            self.create_model(config)
            # possibility to change where we save
            error = self.train_step()
            # print(accuracy, 'accuracy is')

            if np.isnan(error):
                error = 10**5

            self.ITERATION += 1
            return error
        
        search_result = gp_minimize(
            func=fitness,
            dimensions=dimensions,
            acq_func="EI",  # Expected Improvement.
            n_calls=self.n_calls,
            x0=self.default_parameters,
            random_state=1234,
        )

        print(search_result.x)

        plot_convergence(search_result)
        plot_objective(search_result, show_points=True, size=3.8)












