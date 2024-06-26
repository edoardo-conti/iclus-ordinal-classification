import tensorflow as tf
from keras import layers, Model

class CNNConStn(Model):
    def __init__(self, img_size, nclasses, batch_size=32, fixed_scale=True):
        super(CNNConStn, self).__init__()

        self.img_size = img_size
        self.nclasses = nclasses
        self.batch_size = batch_size
        self.fixed_scale = fixed_scale

        self.block1 = tf.keras.Sequential([
            layers.Conv2D(32, (3, 3), strides=1, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(32, (3, 3), strides=1, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=(2, 2), strides=2)
        ])

        self.block2 = tf.keras.Sequential([
            layers.Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=(2, 2), strides=2)
        ])

        self.block3 = tf.keras.Sequential([
            layers.Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=(2, 2), strides=2)
        ])

        self.block4 = tf.keras.Sequential([
            layers.Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=(2, 2), strides=2)
        ])

        self.block5 = tf.keras.Sequential([
            layers.Conv2D(128, (3, 3), strides=1, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(128, (3, 3), strides=1, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=(2, 2), strides=2),
            layers.Dropout(0.3)
        ])

        self.block6 = tf.keras.Sequential([
            layers.Conv2D(128, (3, 3), strides=1, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(128, (3, 3), strides=1, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.ReLU(),
            # nn.AvgPool2d(kernel_size=4)  # paper: 8
        ])

        self.block_out = tf.keras.Sequential([
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(4, activation='relu')
        ])

        if fixed_scale:
            # Regressor for the 3 * 2 affine matrix
            self.fc_loc = tf.keras.Sequential([
                layers.Dense(32, activation='relu'),
                layers.Dense(4, activation=None)  # predict just translation params
            ])
            # Initialize biases
            # self.fc_loc.layers[-1].bias.assign([0.3, 0.3, 0.2, 0.2])
        else:
            # Regressor for the 3 * 2 affine matrix
            self.fc_loc = tf.keras.Sequential([
                layers.Dense(32, activation='relu')
            ])
            self.trans = layers.Dense(4)
            self.scaling = layers.Dense(2)
            self.rotation = layers.Dense(4)

            # Initialize biases
            # self.trans.bias.assign([0.3, 0.3, 0.2, 0.2])
            # self.scaling.bias.assign([0.5, 0.75])
            # self.rotation.bias.assign(tf.random.normal([4], mean=0.0, stddev=0.1))

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
            trans_1, trans_2 = tf.split(trans, num_or_size_splits=2, axis=1)
            # prepare theta for each resolution
            # theta_1 = tf.concat([((tf.eye(2) * 0.5)[:, tf.newaxis] + tf.zeros((2, bs, 1))) , tf.reshape(trans_1, [bs, 2, 1])], axis=2)
            # theta_2 = tf.concat([((tf.eye(2) * 0.75)[:, tf.newaxis] + tf.zeros((2, bs, 1))), tf.reshape(trans_1, [bs, 2, 1])], axis=2)
            # Calcolo di theta_1
            eye_matrix_1 = tf.eye(2) * 0.5
            eye_matrix_1 = tf.expand_dims(eye_matrix_1, axis=0)
            eye_matrix_1 = tf.tile(eye_matrix_1, [bs, 1, 1])
            theta_1 = tf.concat([eye_matrix_1, tf.expand_dims(trans_1, axis=2)], axis=2)

            # Calcolo di theta_2
            eye_matrix_2 = tf.eye(2) * 0.75
            eye_matrix_2 = tf.expand_dims(eye_matrix_2, axis=0)
            eye_matrix_2 = tf.tile(eye_matrix_2, [bs, 1, 1])
            theta_2 = tf.concat([eye_matrix_2, tf.expand_dims(trans_1, axis=2)], axis=2)
        else:
            xs = self.fc_loc(xs)
            # predict the scaling params
            scaling = tf.sigmoid(self.scaling(xs))
            scaling_1, scaling_2 = tf.split(scaling, num_or_size_splits=2, axis=1)
            # predict the translation params
            trans = self.trans(xs)
            bs = tf.shape(trans)[0]
            trans_1, trans_2 = tf.split(trans, num_or_size_splits=2, axis=1)
            # predict the rotation params
            rot = self.rotation(xs)
            rot_1, rot_2 = tf.split(rot, num_or_size_splits=2, axis=1)
            # prepare theta for each resolution
            rot_1 = tf.ones((2, 2)).numpy().diagonal(0).reshape(1, 2, 2).repeat(bs, axis=0) * tf.reshape(rot_1, [bs, 2, 1])
            rot_2 = tf.ones((2, 2)).numpy().diagonal(0).reshape(1, 2, 2).repeat(bs, axis=0) * tf.reshape(rot_2, [bs, 2, 1])
            # add to the scaling params
            rot_1 = rot_1 + tf.eye(2).reshape(1, 2, 2) * tf.reshape(scaling_1, [bs, 1, 1])
            rot_2 = rot_2 + tf.eye(2).reshape(1, 2, 2) * tf.reshape(scaling_2, [bs, 1, 1])
            # prepare the final theta
            theta_1 = tf.concat([rot_1, tf.reshape(trans_1, [bs, 2, 1])], axis=2)
            theta_2 = tf.concat([rot_2, tf.reshape(trans_1, [bs, 2, 1])], axis=2)
        
        # get the shapes
        bs, h, w, c = x.get_shape().as_list()
        stn_out_size = (bs, h, w, c)

        # apply transformations
        grid_1 = self.affine_grid(theta_1, stn_out_size)
        grid_2 = self.affine_grid(theta_2, stn_out_size)

        x_1 = self.grid_sample(x, grid_1)
        x_2 = self.grid_sample(x, grid_2)
        x = tf.concat([x_1, x_2], axis=0)

        return x, scaling

    def call(self, x, training=False):
        x, scaling = self.stn(x)  # transform the input
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        x = self.block_out(x)

        #x_1, x_2 = tf.split(x, num_or_size_splits=2)
        
        #return x_1, x_2

        # return x, scaling 
        return x
    
    def predict(self, x, *args, **kwargs):
        # Personalizza il comportamento di model.predict
        # Ad esempio, puoi aggiungere registrazioni, manipolare i dati in input, ecc.
        predictions = super(CNNConStn, self).predict(x, *args, **kwargs)
        
        # Aggiungi il tuo comportamento personalizzato qui, se necessario
        # Ad esempio, registra le predizioni
        print("Predizioni:", predictions)

        return predictions