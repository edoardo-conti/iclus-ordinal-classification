import tensorflow as tf
from keras import Model, layers, initializers

class CNNStn(Model):
    def __init__(self, img_size, nclasses, batch_size=32, fixed_scale=True):
        super(CNNStn, self).__init__()
        
        self.img_size = img_size
        self.nclasses = nclasses
        self.batch_size = batch_size
        self.fixed_scale = fixed_scale

        # Convolutional Blocks
        self.block1 = tf.keras.Sequential([
            layers.Conv2D(32, (3, 3), strides=1, padding='same', input_shape=(img_size, img_size, 3)),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(32, (3, 3), strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=(2, 2), strides=2)
        ])

        self.block2 = tf.keras.Sequential([
            layers.Conv2D(64, (3, 3), strides=1, padding='same', input_shape=(None, None, 32)),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(64, (3, 3), strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=(2, 2), strides=2)
        ])

        self.block3 = tf.keras.Sequential([
            layers.Conv2D(64, (3, 3), strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(64, (3, 3), strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=(2, 2), strides=2)
        ])

        self.block4 = tf.keras.Sequential([
            layers.Conv2D(64, (3, 3), strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(64, (3, 3), strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=(2, 2), strides=2)
        ])

        self.block5 = tf.keras.Sequential([
            layers.Conv2D(128, (3, 3), strides=1, padding='same', input_shape=(None, None, 64)),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(128, (3, 3), strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=(2, 2), strides=2),
            layers.Dropout(0.3)
        ])

        self.block6 = tf.keras.Sequential([
            layers.Conv2D(128, (3, 3), strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(128, (3, 3), strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

        self.block7 = tf.keras.Sequential([
            layers.Dense(256, input_shape=(128,)),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
        
        self.out = layers.Dense(nclasses, input_shape=(256,))

        # self.block_out = tf.keras.Sequential([
        #     layers.Dense(256, input_shape=(128,)),
        #     layers.BatchNormalization(),
        #     layers.ReLU(),
        #     layers.Dense(4, input_shape=(256,))
        # ])

        if fixed_scale: # scaling is kept fixed, only translation is learned
            # Regressor for the 3 * 2 affine matrix
            self.fc_loc = tf.keras.Sequential([
                layers.Dense(32, input_shape=(128 * 7 * 7,)),
                layers.ReLU(),
                layers.Dense(4, # predict just translation params
                             input_shape=(32,),
                             kernel_initializer='zeros', 
                             bias_initializer=tf.constant_initializer([0.3, 0.3, 0.2, 0.2]))
            ])
        else: # scaling, rotation and translation are learned
            # Regressor for the 3 * 2 affine matrix
            self.fc_loc = tf.keras.Sequential([
                layers.Dense(32, input_shape=(128 * 7 * 7,)),
                layers.ReLU()
            ])
            self.trans = layers.Dense(4, # predict translation params
                                      input_shape=(32,),
                                      kernel_initializer='zeros', 
                                      bias_initializer=tf.constant_initializer([0.3, 0.3, 0.2, 0.2]))
            self.scaling = layers.Dense(2, # predict the scaling parameter
                                        input_shape=(32,),
                                        kernel_initializer='zeros', 
                                        bias_initializer=tf.constant_initializer([0.5, 0.75]))
            self.rotation = layers.Dense(4, # predict the rotation parameters
                                         input_shape=(32,),
                                         kernel_initializer='zeros', 
                                         bias_initializer=initializers.RandomNormal(mean=0, stddev=0.1))
    

    def affine_grid(self, theta, size):
        # Estraiamo la dimensione della griglia
        _, height, width, _  = size

        # Creiamo una griglia di coordinate normalizzate
        x = tf.linspace(-1.0, 1.0, width)
        y = tf.linspace(-1.0, 1.0, height)
        x_t, y_t = tf.meshgrid(x, y)
        ones = tf.ones_like(x_t)
        grid = tf.stack([x_t, y_t, ones], axis=-1)

        # Riformattiamo la griglia per poter fare una moltiplicazione batch-wise
        grid = tf.reshape(grid, [-1, height * width, 3])

        # Applichiamo la trasformazione affine
        theta = tf.reshape(theta, [-1, 2, 3])
        grid = tf.matmul(grid, tf.transpose(theta, [0, 2, 1]))

        # Riportiamo la griglia nella sua forma originale
        grid = tf.reshape(grid, [-1, height, width, 2])

        return grid

    def grid_sample(self, input, grid):
        def process_coord(grid, w_h):
            pixs = (grid + 1) * (0.5 * w_h) - 0.5
            pixs = tf.clip_by_value(pixs, -1, w_h) + 1
            return pixs
        
        def gather(input, y, x, b, h, w, c):
            w_padded = w + 2
            h_padded = h + 2
            
            linear_coordinates = tf.cast(y * w_padded + x, dtype=tf.int32)
            
            #b = tf.size(input) / ( h_padded * w_padded * c )
            b = tf.floor(tf.size(input) / ( w_padded * h_padded * c ))
            
            linear_coordinates = tf.reshape(linear_coordinates, shape=(b, h, w))
            input = tf.reshape(input, shape=(b, h_padded * w_padded, c))
            out = tf.gather(params=input, indices=linear_coordinates, batch_dims=1)
            return out
        
        # b, h, w, c = tf.cast(tf.shape(input), tf.float32)
        b, h, w, c = input.get_shape().as_list()
        b = self.batch_size

        grid_x, grid_y = tf.split(grid, num_or_size_splits=2, axis=-1)
        x = process_coord(grid_x, w)
        y = process_coord(grid_y, h)

        input = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(input)

        x0 = tf.math.floor(x)
        y0 = tf.math.floor(y)
        x1 = tf.math.ceil(x)
        y1 = tf.math.ceil(y)

        dx = x - x0
        dy = y - y0
        oneminus_dx = 1 - dx
        oneminus_dy = 1 - dy
        w_y0_x0 = oneminus_dy * oneminus_dx
        w_y1_x0 = dy * oneminus_dx
        w_y1_x1 = dy * dx
        w_y0_x1 = oneminus_dy * dx

        v_y0_x0 = gather(input, y0, x0, b, h, w, c)
        v_y1_x0 = gather(input, y1, x0, b, h, w, c)
        v_y1_x1 = gather(input, y1, x1, b, h, w, c)
        v_y0_x1 = gather(input, y0, x1, b, h, w, c)

        return w_y0_x0 * v_y0_x0 + w_y1_x0 * v_y1_x0 + w_y1_x1 * v_y1_x1 + w_y0_x1 * v_y0_x1


    # Spatial Transformer Network (STN)
    def stn(self, x):
        scaling = 0  # dummy variable for just translation
        xs = self.block1(x)
        xs = self.block2(xs)
        xs = self.block3(xs)
        xs = self.block4(xs)
        xs = self.block5(xs)
        xs = self.block6(xs)
        xs = tf.reshape(xs, [-1, 128 * 7 * 7])

        if self.fixed_scale:
            trans = self.fc_loc(xs)
            bs = tf.shape(trans)[0]
            trans_1, _ = tf.split(trans, num_or_size_splits=trans.shape[1] // 2, axis=1)

            # prepare theta for each resolution
            eye_2 = tf.eye(2, batch_shape=[bs])
            theta_1 = tf.concat([(eye_2 * 0.5), tf.expand_dims(trans_1, axis=-1)], axis=-1)
            theta_2 = tf.concat([(eye_2 * 0.75), tf.expand_dims(trans_1, axis=-1)], axis=-1)
        else:
            xs = self.fc_loc(xs)
            # predict the scaling params
            scaling = tf.sigmoid(self.scaling(xs))
            scaling_1, scaling_2 = tf.split(scaling, num_or_size_splits=scaling.shape[1] // 2, axis=1)

            # predict the translation params
            trans = self.trans(xs)
            bs = tf.shape(trans)[0]
            trans_1, _ = tf.split(trans, num_or_size_splits=trans.shape[1] // 2, axis=1)
            
            # predict the rotation params
            rot = self.rotation(xs)
            rot_1, rot_2 = tf.split(rot, num_or_size_splits=rot.shape[1] // 2, axis=1)

            # prepare theta for each resolution
            eye_2 = tf.eye(2, batch_shape=[bs])
            rot_1 = eye_2 * 1.0  # Replace with scaling factor for rotation
            rot_2 = eye_2 * 1.0  # Replace with scaling factor for rotation
            rot_1 = rot_1 + tf.eye(2) * scaling_1[:, tf.newaxis, tf.newaxis]
            rot_2 = rot_2 + tf.eye(2) * scaling_2[:, tf.newaxis, tf.newaxis]

            # prepare the final theta
            theta_1 = tf.concat([rot_1, trans_1[:, :, tf.newaxis]], axis=-1)
            theta_2 = tf.concat([rot_2, trans_1[:, :, tf.newaxis]], axis=-1)

        # get the shapes
        bs, h, w, c = x.get_shape().as_list()
        stn_out_size = (bs, h, w, c)

        # apply transformations
        grid_1 = self.affine_grid(theta_1, stn_out_size)
        grid_2 = self.affine_grid(theta_2, stn_out_size)

        # grid sampling
        x_1 = self.grid_sample(x, grid_1)
        x_2 = self.grid_sample(x, grid_2)

        x = tf.concat([x_1, x_2], axis=0)

        return x, scaling


    def call(self, x):
        x, _ = self.stn(x)  # apply the STN
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = self.block7(x)
        x = layers.Dropout(0.3)(x)
        x = self.out(x)

        return x
    

    # # wip
    # def predict(self, x, *args, **kwargs):
    #     # Personalizza il comportamento di model.predict
    #     # Ad esempio, puoi aggiungere registrazioni, manipolare i dati in input, ecc.
    #     predictions = super(CNNStn, self).predict(x, *args, **kwargs)

    #     return predictions