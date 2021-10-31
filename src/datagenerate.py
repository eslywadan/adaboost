import numpy as np
from sklearn.datasets import make_gaussian_quantiles

''' generating a toy dataset using a similar approach to sklearn documentation'''


def dataset(n, random_seed, classes):
    if random_seed:
        np.random.seed(random_seed)
        x, y = make_gaussian_quantiles(n_samples=n, n_features=2, n_classes=classes)
        return x, y * 2 - 1
