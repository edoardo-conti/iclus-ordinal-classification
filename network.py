import tensorflow as tf
import keras
from clm import CumulativeLinkModel

class Net:
    def __init__(self, size, activation, final_activation, num_channels=3,
                num_classes=4, dropout=0, use_tau=True ):
        self.size = size
        self.activation = activation
        self.final_activation = final_activation
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.dropout = dropout
        self.use_tau = use_tau
        
    def build(self, net_model):
        if hasattr(self, net_model):
            return getattr(self, net_model)()
        else:
            raise Exception('Invalid network model.')
    
    def resnet18(self):
        def _resnet_block(x, filters: int, kernel_size=3, init_scheme='he_normal', down_sample=False):
            strides = [2, 1] if down_sample else [1, 1]
            res = x

            x = keras.layers.Conv2D(filters, strides=strides[0], kernel_size=kernel_size, 
                                    padding='same', kernel_initializer=init_scheme)(x)
            x = keras.layers.BatchNormalization()(x)
            x = self.__activation()(x)
            x = keras.layers.Conv2D(filters, strides=strides[1], kernel_size=kernel_size, 
                                    padding='same', kernel_initializer=init_scheme)(x)
            x = keras.layers.BatchNormalization()(x)

            if down_sample:
                # perform down sampling using stride of 2, according to [1].
                res = keras.layers.Conv2D(filters, strides=2, kernel_size=(1, 1),
                                    padding='same', kernel_initializer=init_scheme)(res)

            # if down sampling not enabled, add a shortcut directly 
            x = keras.layers.Add()([x, res])
            out = self.__activation()(x)

            return out
        
        input = keras.layers.Input(shape=(self.size, self.size, self.num_channels))

        x = keras.layers.Conv2D(64, strides=2, kernel_size=(7, 7), 
                                padding='same', kernel_initializer='he_normal')(input)
        x = keras.layers.BatchNormalization()(x)
        x = self.__activation()(x)
        x = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)

        x = _resnet_block(x, 64)
        x = _resnet_block(x, 64)
        x = _resnet_block(x, 128, down_sample=True)
        x = _resnet_block(x, 128)
        x = _resnet_block(x, 256, down_sample=True)
        x = _resnet_block(x, 256)
        x = _resnet_block(x, 512, down_sample=True)
        x = _resnet_block(x, 512)

        x = keras.layers.GlobalAveragePooling2D()(x)

        # TODO: More in-depth studies are needed but it seems to lead the CLM not work
        # x = keras.layers.Flatten()(x)

        if self.dropout > 0:
            x = keras.layers.Dropout(rate=self.dropout)(x)

        x = self.__final_activation(x)

        model = keras.models.Model(input, x)
        
        return model

    def conv128(self):
        feature_filter_size = 3
        classif_filter_size = 4

        def _conv_block(x, filters, kernel_size, max_pooling=False):
            x = keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), kernel_initializer='he_uniform')(x)
            x = self.__activation()(x)
            x = keras.layers.BatchNormalization()(x)
            if max_pooling:
                x = keras.layers.MaxPooling2D()(x)

            return x

        input = keras.layers.Input(shape=(self.size, self.size, self.num_channels))

        x = _conv_block(input, 32, feature_filter_size)
        x = _conv_block(x, 32, feature_filter_size, True)

        x = _conv_block(x, 64, feature_filter_size)
        x = _conv_block(x, 64, feature_filter_size, True)

        x = _conv_block(x, 128, feature_filter_size)
        x = _conv_block(x, 128, feature_filter_size, True)

        x = _conv_block(x, 128, feature_filter_size)
        x = _conv_block(x, 128, feature_filter_size, True)

        x = _conv_block(x, 128, classif_filter_size)

        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(96)(x)

        if self.dropout > 0:
            x = keras.layers.Dropout(rate=self.dropout)(x)

        x = self.__final_activation(x)

        model = keras.models.Model(input, x)
        
        return model

    def __activation(self):
        if self.activation == 'lrelu':
            return keras.layers.LeakyReLU()
        elif self.activation == 'prelu':
            return keras.layers.PReLU()
        elif self.activation == 'elu':
            return keras.layers.ELU()
        else:
            return keras.layers.Activation('relu')
    
    def __final_activation(self, x):
        final_activations_dict = {
            'poml': 'logit',
            'pomp': 'probit',
            'pomc': 'cloglog',
            'pomg': 'glogit',
        }
        
        activation = final_activations_dict.get(self.final_activation, 'softmax')

        if activation == 'softmax':
            x = keras.layers.Dense(self.num_classes)(x)
            x = keras.layers.Activation(self.final_activation)(x)
        else:
            x = keras.layers.Dense(1)(x)
            x = keras.layers.BatchNormalization()(x)
            x = CumulativeLinkModel(self.num_classes, activation, use_tau=self.use_tau)(x)

        return x