# Market simulator

Code for the paper:

> A Data-driven Market Simulator for Small Data Environments, https://arxiv.org/abs/2006.14498, H. Buhler, B. Horvath, T. Lyons, I. Perez Arribas and B. Wood.

Source code:
> https://github.com/imanolperez/market_simulator/tree/master

## Problems of the Source Code
1. Performance considerations:
    - Too old versions: The deep learning model in the source code is mainly written by tensorflow 1.15, released about 4 years ago. If you want to run the source code, you have to download the old tensorflow and its related dependencies. It's very likely you encounter annoying bugs.
    - Python-based (Numpy-based) calculations: Most of the calculations of signature are powered by python, which is much slower than cpp-based library. For large datasets, the functions provided in the source code are not a good choice for calculations.
    - Reproducibility: The code author does not fix the random seed in his code. Moreover, even with exactly the same model and hyperparameter, the performance of my model is terrible, unlike the results in the source code. 

2. Bugs in self-defined functions

## Understanding Market Simulator
Here is my [notebook](https://colab.research.google.com/drive/1d-_oFr7ypn8n3XW_v9QbJaiHw6_rUfEk#scrollTo=VzLqr_en9VtA) (not finished) for explaning the idea and structure of market simulator, including its motivation, background of VAE and the numerical results (still reproducing...). 