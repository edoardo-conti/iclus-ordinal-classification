import keras
import numpy as np
from keras import backend as K
import tensorflow as tf

def make_cost_matrix(num_ratings):
	"""
	Create a quadratic cost matrix of num_ratings x num_ratings elements.
     
	:param num_ratings: number of labels.
	:return: cost matrix.
	"""

	cost_matrix = np.reshape(np.tile(range(num_ratings), num_ratings), (num_ratings, num_ratings))
	cost_matrix = np.power(cost_matrix - np.transpose(cost_matrix), 2) / (num_ratings - 1) ** 2.0
	return np.float32(cost_matrix)

def qwk_loss(cost_matrix):
    def _qwk_loss(y_true, y_pred):
        targets = K.argmax(y_true, axis=1)
        costs = K.gather(cost_matrix, targets)

        numerator = costs * y_pred
        numerator = K.sum(numerator)

        sum_prob = K.sum(y_pred, axis=0)
        n = K.sum(y_true, axis=0)  

        a = tf.cast(K.reshape(K.dot(cost_matrix, K.reshape(sum_prob, shape=[-1, 1])), shape=[-1]), dtype=tf.float32)
        b = tf.cast(K.reshape(n / K.sum(n), shape=[-1]), dtype=tf.float32)

        epsilon = 10e-9

        denominator = a * b
        denominator = K.sum(denominator) + epsilon

        return numerator / denominator

    return _qwk_loss