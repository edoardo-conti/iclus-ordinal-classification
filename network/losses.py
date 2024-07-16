import numpy as np
from keras import backend as K
import tensorflow as tf

def ordinal_distance_loss(n_classes):
    target_class = tf.ones((n_classes, n_classes - 1), dtype=tf.float32)
    target_class = 1 - tf.linalg.band_part(target_class, 0, -1) 
    '''
    Example: target_class with num_classes = 4 -> (4, 3)
    [
        [0., 0., 0.],
        [1., 0., 0.],
        [1., 1., 0.],
        [1., 1., 1.]
    ]
    '''
    
    mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
    
    def _ordinal_distance_loss(y_true, y_pred): 
        indices = tf.argmax(y_true, axis=1)
        y_true = tf.gather(target_class, indices)
        
        return mse(y_pred, y_true)

    return _ordinal_distance_loss


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


def cnnstn_cce_loss():
    cce = tf.keras.losses.CategoricalCrossentropy()
    
    def _cnnstn_cce_loss(y_true, y_pred):
        y_pred_0, _ = tf.split(y_pred, num_or_size_splits=2, axis=0)
        y_pred_0 = tf.nn.softmax(y_pred_0, axis=1)
        
        return cce(y_true, y_pred_0)

    return _cnnstn_cce_loss


def sord_loss(nn_model, cost_mul=2, lambda_reg=1.):    
    def _sord_loss(y_true, y_pred):
        # get the actual batch_size (and classes)
        batch_size = tf.shape(y_true)[0]
        num_classes = tf.shape(y_true)[1]
        
        # creating labels_sord with effective batch size x classes
        labels_sord = tf.zeros([batch_size, num_classes], tf.int32)
        
        # build the SORD label
        for batch_idx in range(batch_size):
            current_label = tf.argmax(y_true[batch_idx], axis=0, output_type=tf.int32)
            for class_idx in range(num_classes):
                value = tf.cast(cost_mul, tf.int32) * tf.math.square(tf.abs(current_label - class_idx))
                labels_sord = tf.tensor_scatter_nd_update(labels_sord, [[batch_idx, class_idx]], [value])

        # split the operations based on neural network used
        if nn_model == 'cnnstn':
            # split the network output if CNNStn
            y_pred, y_pred_2 = tf.split(y_pred, num_or_size_splits=2, axis=0)
            # consistency loss
            mse_loss = lambda_reg * tf.reduce_mean(tf.square(y_pred - y_pred_2))
        
        # compute the prediction using log softmax for numerical stability
        log_predictions_probs = tf.nn.log_softmax(y_pred, axis=1)

        # compute the softmax for the SORD labels
        labels_sord_probs = tf.nn.softmax(tf.cast(-labels_sord, tf.float32), axis=1)
        
        # cross entropy
        sord_loss = tf.reduce_mean(tf.reduce_sum(-labels_sord_probs * log_predictions_probs, axis=1))
        
        # assign the final loss return value
        loss = sord_loss + mse_loss if nn_model == 'cnnstn' else sord_loss

        return loss

    return _sord_loss