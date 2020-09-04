import tensorflow as tf
import numpy as np
# import networkx as nx

from gat import *

A = tf.constant([
    [0, 1, 0, 0],
    [0, 0, 1, 1],
    [0, 1, 0, 0],
    [1, 0, 1, 0]], dtype=tf.float32)

X = np.array([
    [i, -i] for i in range(tf.shape(A)[0])
], dtype=float)

X = tf.cast(X, tf.float32)

gat = GraphAttentionLayer(8, 8)

print(gat([X, A]))