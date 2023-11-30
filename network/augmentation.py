import tensorflow as tf

class USDataAugmentation():
    def __init__(self, input_size=224, random_state=42):
        self.input_size = input_size
        self.random_state = random_state

    def build(self):
        tf.random.set_seed(self.random_state)
        
        self._random_rotation = tf.keras.layers.RandomRotation(factor=0.05, seed=self.random_state)

    def _central_crop(self, frame):
        random_zoom = tf.random.uniform([], minval=0.7, maxval=0.9, dtype=tf.float32)
        frame_cropped = tf.image.central_crop(frame, central_fraction=random_zoom)
        frame_cropped = tf.image.resize(frame_cropped, [self.input_size, self.input_size])
        return frame_cropped

    def _adjust_brightness(self, frame):  
        random_delta = tf.random.uniform([], minval=-0.1, maxval=0.15, dtype=tf.float32)
        frame_bright = tf.image.adjust_brightness(frame, random_delta)
        frame_bright = tf.clip_by_value(frame_bright, 0.0, 1.0)
        return frame_bright
    
    def _random_true_false(self):
        return tf.less(tf.random.uniform([], 0., 1.), 0.5)

    def _random_flip_left_right(self, frame):
        prob = self._random_true_false()
        frame_aug = tf.cond(prob, lambda: tf.image.flip_left_right(frame), lambda: frame)
        return frame_aug

    def _random_flip_up_down(self, frame):
        prob = self._random_true_false()
        frame_aug = tf.cond(prob, lambda: tf.image.flip_up_down(frame), lambda: frame)
        return frame_aug

    def _random_centeral_crop(self, frame):  
        prob = self._random_true_false()
        frame_aug = tf.cond(prob, lambda: self._central_crop(frame), lambda: frame)
        return frame_aug

    def _random_adjust_brightness(self, frame):  
        prob = self._random_true_false()
        frame_aug = tf.cond(prob, lambda: self._adjust_brightness(frame), lambda: frame)
        return frame_aug

    def us_augmentation(self, frame):
        frame = self._random_flip_left_right(frame)
        frame = self._random_centeral_crop(frame)
        frame = self._random_adjust_brightness(frame)
        frame = self._random_rotation(frame)

        return frame