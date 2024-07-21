from sklearn.manifold import TSNE
import numpy as np

# ----- dimension reduction by T-SNE
from utility import TSNE_to_2D

# [3, 5]
input = np.random.randint(0, 10, (3,5))

# [3, 2]
input_2d = TSNE_to_2D(input)