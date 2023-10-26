import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix

class MyMetrics:
    def __init__(self):
        # ======= OBD =======
        self.target_class = np.ones((4, 4 - 1), dtype=np.float32)
        self.target_class[np.triu_indices(4, 0, 4 - 1)] = 0.0

    def from_obdout_to_labels(self, obd_y_pred):
        # Calculate pairwise distances between y_pred and target_class using Euclidean distance
        distances = tf.norm(tf.expand_dims(obd_y_pred, 1) - self.target_class, axis=2, ord='euclidean')
        
        # Find the index of the minimum distance as the predicted label
        labels = tf.argmin(distances, axis=1)

        return labels

    def ccr(self, y_true, y_pred):
        # predit labels for each sample from network output
        y_pred = self.from_obdout_to_labels(y_pred)

        # Find the true labels
        y_true = tf.argmax(y_true, axis=-1) 

        # Check for correct predictions
        correct = tf.equal(y_true, y_pred)

        # Calculate the Correct Classification Rate
        ccr_value = tf.reduce_mean(tf.cast(correct, tf.float32))

        return ccr_value

    def ms(self, y_true, y_pred):
        # !NOT SURE IF WORKING!
        # predit labels for each sample from network output
        y_pred = self.from_obdout_to_labels(y_pred)

        # Find the true labels
        y_true = tf.argmax(y_true, axis=-1) 

        # Calculate confusion matrix
        #cm = confusion_matrix(y_true, y_pred).astype(float)
        cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=4, dtype=tf.float32)
        
        # Calculate sum of true positives (TP) for each class
        #sum_byclass = np.sum(cm, axis=1).astype(float)
        sum_byclass = tf.reduce_sum(cm, axis=1)

        # Calculate sensitivities (TP / sum by class)
        #sensitivities = np.diag(cm) / sum_byclass
        sensitivities = tf.linalg.diag_part(cm) / sum_byclass

        # Find the minimum sensitivity
        ms = tf.reduce_min(sensitivities)

        return ms

    def mae(self, y_true, y_pred):
        # predit labels for each sample from network output
        y_pred = self.from_obdout_to_labels(y_pred)

        # Find the true labels
        y_true = tf.argmax(y_true, axis=-1) 

        y_true = tf.convert_to_tensor(y_true, dtype=tf.int64)
        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.int64)

        absolute_errors = tf.abs(y_true - y_pred)
        mean_absolute_error = tf.reduce_sum(absolute_errors) / tf.cast(tf.shape(y_true)[0], dtype=tf.int64)

        return mean_absolute_error

    def accuracy_oneoff(self, y_true, y_pred):
        # predit labels for each sample from network output
        y_pred = self.from_obdout_to_labels(y_pred)

        #if y_true.shape[0] > 1:
        # Find the true labels
        y_true = tf.argmax(y_true, axis=-1) 

        conf_mat = tf.math.confusion_matrix(y_true, y_pred, num_classes=4, dtype=tf.float32)
        n = tf.shape(conf_mat)[0]

        #mask = tf.eye(n) + tf.eye(n, delta=1) + tf.eye(n, delta=-1)
        mask = tf.linalg.diag(tf.ones(n)) + tf.linalg.diag(tf.ones(n - 1), k=1) + tf.linalg.diag(tf.ones(n - 1), k=-1)
        correct = mask * conf_mat
        
        accuracy_1off = tf.reduce_sum(correct) / tf.reduce_sum(conf_mat)

        return accuracy_1off