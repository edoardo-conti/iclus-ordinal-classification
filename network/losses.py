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


def sord_loss(net_type, num_classes=4, multiplier=2, wide_gap_loss=False):
    def _sord_loss(ground_truth, logits):
        batch_size = tf.shape(ground_truth)[0]  # Ottieni il batch size come tensore simbolico
        
        labels_sord = tf.TensorArray(tf.float32, size=batch_size)
        for element_idx in tf.range(batch_size):
            
            current_label = tf.cast(ground_truth[element_idx], tf.float32)
            class_indices = tf.range(num_classes, dtype=tf.float32)

            # Calcola le distanze tra l'etichetta reale e ogni classe prevista
            if wide_gap_loss:
                wide_label = tf.cond(tf.equal(current_label, 0), lambda: -0.5, lambda: current_label)
                wide_class_indices = tf.cond(tf.equal(class_indices, 0), lambda: -0.5, lambda: class_indices)
                distances = multiplier * tf.square(tf.abs(wide_label - wide_class_indices))
            else:
                distances = multiplier * tf.square(tf.abs(current_label - class_indices))
            
            labels_sord = labels_sord.write(element_idx, distances)
        
        labels_sord = labels_sord.stack()
        labels_sord = tf.nn.softmax(-labels_sord, axis=1)
        
        # Predizioni logaritmiche
        if net_type == 'roysota':
            output_1, output_2 = tf.split(logits, num_or_size_splits=2, axis=0)
            log_predictions = tf.nn.log_softmax(output_1, axis=1)
            
            # output_1 = logits[0]
            # output_2 = logits[1]
            # log_predictions = tf.nn.log_softmax(output_1, axis=-1)
        else:
            log_predictions = tf.nn.log_softmax(logits, axis=1)

        # Calcola la loss cross-entropy
        loss = -tf.reduce_sum(labels_sord * log_predictions, axis=1)
        loss = tf.reduce_mean(loss)
        
        # aggiungo la consistency loss MSE
        if net_type == 'roysota':
            # Calcola la media del quadrato della differenza
            mse_loss = 0.5 * tf.reduce_mean(tf.square(output_1 - output_2))
            loss = loss + mse_loss
        
        # tf.print("ground_truth[0]", ground_truth[0])
        # tf.print("log_predictions[0]", log_predictions[0])
        
        # tf.print("logits[0]", logits[0])
        # tf.print("output_1[0]", output_1[0])
        # tf.print("output_2[0]", output_2[0])

        return loss

    return _sord_loss


def roy_cce_loss():
    def _roy_cce_loss(y_true, y_pred):
        #y_pred_1, _ = tf.split(y_pred, num_or_size_splits=2)

        cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

        return cce(y_true, y_pred[0])

    return _roy_cce_loss