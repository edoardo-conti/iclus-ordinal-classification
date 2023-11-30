import keras
import tensorflow as tf
import tensorflow_probability
import tensorflow as tf

class CumulativeLinkModel(keras.layers.Layer):
	"""
	Proportional Odds Model activation layer.
	"""
	def __init__(self, num_classes, link_function, use_tau, **kwargs):
		super(CumulativeLinkModel, self).__init__(**kwargs)
		self.num_classes = num_classes
		self.link_function = link_function
		self.use_tau = use_tau
		self.dist = tensorflow_probability.distributions.Normal(loc=0., scale=1.)

	def _convert_thresholds(self, b, a):
		a = tf.pow(a, 2)
		thresholds_param = tf.concat([b, a], axis=0)
		ones_matrix = tf.linalg.band_part(tf.ones([self.num_classes - 1, self.num_classes - 1]), -1, 0)
		thresholds_param_tiled = tf.tile(thresholds_param, [self.num_classes - 1])
		reshaped_param = tf.reshape(thresholds_param_tiled, shape=[self.num_classes - 1, self.num_classes - 1])
		th = tf.reduce_sum(ones_matrix * reshaped_param, axis=1)

		return th

	def _nnpom(self, projected, thresholds):
		if self.use_tau == 1:
			projected = tf.reshape(projected, shape=[-1]) / self.tau
		else:
			projected = tf.reshape(projected, shape=[-1])

		m = tf.shape(projected)[0]
		a = tf.reshape(tf.tile(thresholds, [m]), shape=[m, -1])
		b = tf.transpose(tf.reshape(tf.tile(projected, [self.num_classes - 1]), shape=[-1, m]))
		z3 = a - b

		if self.link_function == 'probit':
			a3T = self.dist.cdf(z3)
		elif self.link_function == 'cloglog':
			a3T = 1 - tf.exp(-tf.exp(z3))
		elif self.link_function == 'glogit':
			a3T = 1.0 / tf.pow(1.0 + tf.exp(-self.lmbd * (z3 - self.mu)), self.alpha)
		else:  # 'logit'
			a3T = 1.0 / (1.0 + tf.exp(-z3))
		
		a3 = tf.concat([a3T, tf.ones([m, 1])], axis=1)
		a3 = tf.concat([tf.reshape(a3[:, 0], shape=[-1, 1]), a3[:, 1:] - a3[:, :-1]], axis=-1)

		return a3

	def build(self, input_shape):
		self.thresholds_b = self.add_weight('b_b_nnpom', shape=(1,), trainable=True,
											initializer=tf.keras.initializers.RandomUniform(minval=0, maxval=0.1))
		self.thresholds_a = self.add_weight('b_a_nnpom', shape=(self.num_classes - 2,), trainable=True,
											initializer=tf.keras.initializers.RandomUniform(
											minval=tf.sqrt((1.0 / (self.num_classes - 2)) / 2),
											maxval=tf.sqrt(1.0 / (self.num_classes - 2))))

		if self.use_tau == 1:
			self.tau = self.add_weight('tau_nnpom', shape=(1,),
										initializer=tf.keras.initializers.RandomUniform(minval=1, maxval=10))
			self.tau = tf.clip_by_value(self.tau, 1, 1000)

		if self.link_function == 'glogit':
			self.lmbd = self.add_weight('lambda_nnpom', shape=(1,),
										initializer=tf.keras.initializers.RandomUniform(minval=1, maxval=1))
			self.alpha = self.add_weight('alpha_nnpom', shape=(1,),
										initializer=tf.keras.initializers.RandomUniform(minval=1, maxval=1))
			self.mu = self.add_weight('mu_nnpom', shape=(1,),
										initializer=tf.keras.initializers.RandomUniform(minval=0, maxval=0))
	
	def call(self, x, **kwargs):
		self.thresholds = self._convert_thresholds(self.thresholds_b, self.thresholds_a)
		
		return self._nnpom(x, self.thresholds)

	def compute_output_shape(self, input_shape):
		return (input_shape[0], 1)