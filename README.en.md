# AI4XDE

#### Description
AI4XDE is a comprehensive library for scientific machine learning and physical information networks. AI4XDE aims to decouple specific algorithms from specific examples, using examples as input parameters for neural networks, so that all examples can be calculated in one programming operation. Writing neural network algorithms and examples according to the interface paradigm used in the AI4XDE library can quickly test the stability of algorithms on different examples and accelerate experimental progress; At the same time, it can also enable the completion of calculation examples, which can be tested and compared on different neural network algorithms.

Currently, AI4XDE supports the following algorithms:

1. PINN
2. Uniform
3. Random_ R
4. RAR_ D
5. RAR_ G
6. RAD
7. R3Sampling
8. HPO
9. gPINN

Currently,  AI4XDE supports the following examples:

1. Formula based approximate function calculation example
2. Data based formula approximation examples
3. A simple ODE calculation example
4. Lotka Volterra equation
5. Second Order ODE
6. Poisson equation in 1D with Dirichlet boundary conditions
7. Poisson equation in 1D with Dirichlet/Neumann boundary conditions
8. Poisson equation in 1D with Dirichlet/Robin boundary conditions
9. Poisson equation in 1D with Dirichlet/Periodic boundary conditions
10. Poisson equation in 1D with Dirichlet/PointSetOperator boundary conditions
11. Poisson equation in 1D with hard boundary conditions
12. Poisson equation in 1D with Multi-scale Fourier feature networks
13. Poisson equation over L-shaped domain
14. Inverse problem for the Poisson equation with unknown forcing field
15. Inverse problem for the fractional Poisson equation in 1D
16. Inverse problem for the fractional Poisson equation in 2D
17. Laplace equation on a disk
18. Euler Beam
19. Helmholtz equation over a 2D square domain
20. Helmholtz equation over a 2D square domain with a hole
21. Helmholtz sound-hard scattering problem with absorbing boundary conditions
22. Kovasznay Flow
23. Burgers equation
24. Heat equation
25. Diffusion equation
26. Diffusion-reaction equation
27. Allen Cahn equation
28. Klein-Gordon equation
29. Beltrami flow
30. Schrodinger equation
31. Wave propagation with spatio-temporal multi-scale Fourier feature architecture
32. Wave equation
33. Integro-differential equation
34. Volterra IDE
35. Fractional Poisson equation in 1D
36. Fractional Poisson equation in 2D
37. Fractional Poisson equation in 3D
38. Fractional_Diffusion_1D
39. Inverse problem for the Lorenz system
40. Inverse problem for the Lorenz system with exogenous input
41. Inverse problem for Brinkman-Forchheimer model
42. Inverse problem for the diffusion equation
43. Inverse problem for the Diffusion-reaction equation
44. Inverse problem for the Navier-Stokes equation of incompressible flow around cylinder

#### Installation

Since AI4XDE is based on the DeepXDE library, you need to first install the DeepXDE library.

DeepXDE requires one of the following dependencies to be installed:

- TensorFlow 1.x: [TensorFlow](https://www.tensorflow.org/)>=2.7.0
- TensorFlow 2.x: [TensorFlow](https://www.tensorflow.org/)>=2.2.0, [TensorFlow Probability](https://www.tensorflow.org/probability)>=0.10.0
- PyTorch: [PyTorch](https://pytorch.org/)>=1.9.0
- JAX: [JAX](https://jax.readthedocs.io/), [Flax](https://flax.readthedocs.io/), [Optax](https://optax.readthedocs.io/)
- PaddlePaddle: [PaddlePaddle](https://www.paddlepaddle.org.cn/en) ([develop version](https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/develop/install/pip/linux-pip_en.html))

Please install the above dependencies as a baseline before installing DeepXDE

Subsequently, you can use the following method to install AI4XDE

- Install using 'pip':

```
$ pip install ai4xde
```
- Install using 'conda':
```
$ conda install -c xuelanghanbao ai4xde
```
- For developers, you should clone the folder to your local machine and put it along with your project scripts:
```
$ git clone https://gitee.com/xuelanghanbao/AI4XDE.git
```

#### Instructions

AI4XDE separates algorithms from examples, where algorithm templates are stored in the `solver` folder, and specific algorithms implemented based on algorithm templates (such as PINN, R3Sampling, etc.) are stored in the `algorithms` folder. The calculation template and specific calculation examples (such as Burgers, AllenCahn, etc.) are stored in the `cases` folder.

#### Contribution

1.  Fork the repository
2.  Create Feat_xxx branch
3.  Commit your code
4.  Create Pull Request
