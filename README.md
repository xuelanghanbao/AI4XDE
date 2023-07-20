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
8. HPO
9. gPINN

目前AI4XDE支持算例如下：

1. 基于公式的函数近似算例
2. 基于数据的公式近似算例
3. 一个简单的ODE算例
4. Lotka-Volterra方程
5. 一个二阶ODE算例
6. 具有狄利克雷边界条件的一维Poisson方程
7. 具有狄利克雷/纽曼边界条件的一维Poisson方程
8. 具有狄利克雷/罗宾边界条件的一维Poisson方程
9. 具有狄利克雷/周期边界条件的一维Poisson方程
10. 具有狄利克雷/点集算子边界条件的一维Poisson方程
11. 具有强制边界条件的一维Poisson方程
12. 具有多尺度傅立叶特征网络的一维Poisson方程
13. L型区域上的二维Poisson方程
14. 具有未知强迫场的Poisson方程反问题
15. 圆型区域上的Laplace方程
16. 欧拉梁
17. 二维矩形区域上的Helmholtz方程
18. 带孔二维矩形区域上的Helmholtz方程
19. 具有吸收边界条件的Helmholtz声硬散射问题
20. Kovasznay流
21. Burgers方程
22. Heat方程
23. Diffusion方程
24. Diffusion-reaction方程
25. AllenCahn方程
26. Klein-Gordon方程
27. Beltrami流
28. 薛定谔方程
29. 具有强制边界条件的一维Wave方程
30. 使用时空多尺度傅里叶特征网络的一维Wave方程
31. 积分微分方程
32. Volterra积分微分方程
33. 一维分数阶Poisson方程
34. 二维分数阶Poisson方程
35. 三维分数阶Poisson方程
36. 一维分数阶Diffusion方程
37. Lorenz反问题
38. 具有外生输入的Lorenz反问题
39. Brinkman-Forchheimer反问题
40. Diffusion反问题
41. Diffusion-Reaction反问题


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
