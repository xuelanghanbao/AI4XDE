# AI4XDE

#### 介绍
AI4XDE是一个用于科学机器学习和物理信息网络的综合库。AI4XDE旨在将具体算法与具体算例相解耦，将算例作为神经网络的输入参数，使得一次编程即可计算所有的算例。按照AI4XDE库中使用的接口范式编写神经网络算法以及算例，可以快速地测试算法在不同算例上的稳定性，加快实验进度；同时也可以使算例编写完成，即可在不同的神经网络算法上进行测试、比较。

目前AI4XDE支持算法如下：

1. PINN
2. Uniform
3. Random_R
4. RAR_D
5. RAR_G
6. RAD
7. R3Sampling

目前AI4XDE支持算例如下：

1. 基于公式的函数近似算例
2. 基于数据的公式近似算例
3. Burgers
4. AllenCahn
5. Diffusion
6. Wave
7. Diffusion_Reaction_Inverse
8. 一个简单的ODE算例
9. Lotka-Volterra


#### 安装教程

由于AI4XDE基于DeepXDE库，所以你需要首先安装DeepXDE库。

DeepXDE需要安装以下依赖项之一:

- TensorFlow 1.x: [TensorFlow](https://www.tensorflow.org/)>=2.7.0
- TensorFlow 2.x: [TensorFlow](https://www.tensorflow.org/)>=2.2.0, [TensorFlow Probability](https://www.tensorflow.org/probability)>=0.10.0
- PyTorch: [PyTorch](https://pytorch.org/)>=1.9.0
- JAX: [JAX](https://jax.readthedocs.io/), [Flax](https://flax.readthedocs.io/), [Optax](https://optax.readthedocs.io/)
- PaddlePaddle: [PaddlePaddle](https://www.paddlepaddle.org.cn/en) ([develop version](https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/develop/install/pip/linux-pip_en.html))

请安装完上述依赖项作为基线后，再安装DeepXDE.

随后，你就可以使用如下方法安装AI4XDE

- 使用 `pip`安装：

```
$ pip install ai4xde
```

- 使用 `conda`安装：

```
$ conda install -c xuelanghanbao ai4xde
```

- 对于开发者， 可以将其克隆到本地计算机上：

```
$ git clone https://gitee.com/xuelanghanbao/AI4XDE.git
```

#### 使用说明

AI4XDE将算法与算例分离，其中算法模板存放在 `solver` 文件夹中，基于算法模板实现的具体算法（如：PINN、R3Sampling等）存放在`algorithm`文件夹中。算例模板及具体算例（如：Burgers、AllenCahn等）存放在`cases`文件夹中。

#### 参与贡献

1.  Fork 本仓库
2.  新建 Feat_xxx 分支
3.  提交代码
4.  新建 Pull Request
