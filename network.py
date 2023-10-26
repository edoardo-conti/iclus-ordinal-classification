import tensorflow as tf
import keras
import numpy as np
from clm import CumulativeLinkModel
from keras.applications.vgg16 import VGG16

class Net:
    def __init__(self, 
                 ds_img_size:int = 224,
                 ds_num_ch:int = 3,
                 ds_num_classes:int = 4,
                 nn_dropout:float = 0.0,
                 nn_activation:str = None,
                 nn_final_activation:str = None,
                 clm_use_tau:bool = None,
                 hidden_size:int = None):
        self.size = ds_img_size
        self.activation = nn_activation
        self.final_activation = nn_final_activation
        self.num_channels = ds_num_ch
        self.num_classes = ds_num_classes
        self.dropout = nn_dropout
        self.use_tau = clm_use_tau
        self.hidden_size = hidden_size
        
    def build(self, net_model):
        if hasattr(self, net_model):
            return getattr(self, net_model)()
        else:
            raise Exception('Invalid network model.')
    
    def conv_vgg16(self, input_shape, weights=None, frozen=True, end_pooling='avg'):
        # Define the convolutional VGG16 part
        vgg16 = VGG16(include_top=False, weights=weights, input_shape=input_shape, pooling=end_pooling)
        
        # Freeze the convolutional layers
        if frozen:
            for layer in vgg16.layers[:-1]:
                layer.trainable = False

        return vgg16

    def dense_obd(self, x, hidden_size):
        # local settings
        hidden_size_per_unit = np.round(hidden_size / (self.num_classes - 1)).astype(int)
        
        # define layers
        dense_hidden = [keras.layers.Dense(hidden_size_per_unit) for _ in range(self.num_classes - 1)]
        dense_dropout = [keras.layers.Dropout(self.dropout) for _ in range(self.num_classes - 1)]
        dense_output = [keras.layers.Dense(1) for _ in range(self.num_classes - 1)]
        #dense_bn = keras.layers.BatchNormalization()
        dense_lrelu = keras.layers.LeakyReLU()
        dense_sigmoid = keras.layers.Activation('sigmoid')

        xs = [drop(dense_lrelu(hidden(x))) for hidden, drop in zip(dense_hidden, dense_dropout)]
        xs = [dense_sigmoid(output(xc))[:, 0] for output, xc in zip(dense_output, xs)]

        out = tf.concat([tf.expand_dims(xc, axis=1) for xc in xs], axis=1)

        return out
    
    def obd(self):
        input_shape = (self.size, self.size, self.num_channels)
        hidden_size = 512

        vgg16_conv = self.conv_vgg16(input_shape, 'vgg16_imagenet_notop.h5')
        obd_dense = self.dense_obd(vgg16_conv.output, hidden_size)

        model = keras.models.Model(vgg16_conv.input, obd_dense)

        return model

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

        # TODO: More in-depth studies are needed but it seems that it makes the CLM no longer working.
        #       Therefore at the moment it is left active only in the nominal case.
        if self.final_activation == 'softmax':
            x = keras.layers.Flatten()(x)

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
            x = keras.layers.Dense(self.num_classes, activation=activation)(x)
            #x = keras.layers.Dense(self.num_classes)(x)
            #x = keras.layers.Activation(self.final_activation)(x)
        else:
            x = keras.layers.Dense(1)(x)
            x = keras.layers.BatchNormalization()(x)
            x = CumulativeLinkModel(self.num_classes, activation, use_tau=self.use_tau)(x)

        return x