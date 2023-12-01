# Market simulator

Code for the paper:

> A Data-driven Market Simulator for Small Data Environments, https://arxiv.org/abs/2006.14498, H. Buhler, B. Horvath, T. Lyons, I. Perez Arribas and B. Wood.

Source code:
> https://github.com/imanolperez/market_simulator/tree/master

**Not finished yet!**

## Problems of the Source Code
1. Performance considerations:
    - Too old versions: The deep learning model in the source code is mainly written by tensorflow 1.15, released about 4 years ago. If you want to run the source code, you have to download the old tensorflow and its related dependencies. It's very likely you encounter annoying bugs.
    - Python-based (Numpy-based) calculations: Most of the calculations of signature are powered by python, which is much slower than cpp-based library. For large datasets, the functions provided in the source code are not a good choice for calculations.
    - Reproducibility: The code author does not fix the random seed in his code. Moreover, even with exactly the same model and hyperparameter, the performance of my model is terrible, unlike the results in the source code. I put lots of effort into tuning hyperparameter and adjusting my model.

2. Bugs in self-defined functions
    - While applying the transform of logsignature to signature, the index error popped out, coming from the incorrect indexing of hall basis element, which traces back to tjl_dense_numpy_tensor.py and tjl_hall_numpy_lie.py, the most confusing part of code. Some functions can be checkedd manually, but the majority of the code is difficult to check because of poor readability. The code author does not put much work on explanations while introducing informal terms. It's quite hard to follow the code author's thoughts, and hence, I had a difficult time of debugging and verifying those self-defined functions. Therefore, I changed to [signatory](https://github.com/patrick-kidger/signatory), much faster and easily used.

## Understanding Market Simulator
Here is my [notebook](https://colab.research.google.com/drive/1d-_oFr7ypn8n3XW_v9QbJaiHw6_rUfEk#scrollTo=VzLqr_en9VtA) (not finished) for explaning the idea and structure of market simulator, including its motivation, background of VAE and the numerical results (still reproducing...). 