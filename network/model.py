import numpy as np
import tensorflow as tf
import keras
from keras.applications.vgg16 import VGG16
from network.clm import CumulativeLinkModel

class NeuralNetwork:
    def __init__(self, 
                 ds_img_size:int = 224,
                 ds_img_channels:int = 3,
                 ds_num_classes:int = 4,
                 nn_backbone:str = '',
                 nn_dropout:float = 0.0,
                 nn_activation:str = 'relu',
                 clm_link:str = 'logit',
                 clm_use_tau:bool = True,
                 obd_hidden_size:int = 512):
        self.size = ds_img_size
        self.num_channels = ds_img_channels
        self.num_classes = ds_num_classes
        self.nn_backbone = nn_backbone
        self.dropout = nn_dropout
        self.activation = nn_activation
        self.clm_link = clm_link
        self.use_tau = clm_use_tau
        self.obd_hidden_size = obd_hidden_size

        # computed params
        self.input_shape = (self.size, self.size, self.num_channels)
        
    def build(self, net_model):
        if hasattr(self, net_model):
            return getattr(self, net_model)()
        else:
            raise Exception('Invalid network model.')

    def __activation(self):
        if self.activation == 'lrelu':
            return keras.layers.LeakyReLU()
        elif self.activation == 'prelu':
            return keras.layers.PReLU()
        elif self.activation == 'elu':
            return keras.layers.ELU()
        else:
            return keras.layers.Activation('relu')
    
    def __nominal_final_activation(self, x, final_act='softmax'):
        # apply the dropout layer if requested
        if self.dropout > 0:
            x = keras.layers.Dropout(rate=self.dropout)(x)

        # add the final dense layer with the softmax activation
        x = keras.layers.Dense(self.num_classes)(x)
        x = keras.layers.Activation(final_act)(x)

        return x    

    def _vgg16_convnet(self, input_shape, weights=None, frozen=True, end_pooling='avg'):
        # Define the convolutional VGG16 part
        vgg16 = VGG16(include_top=False, weights=weights, input_shape=input_shape, pooling=end_pooling)
        
        # When performing transfer learning freeze only the convolutional layers and leave
        # trainable the last pooling layer specified with the 'end_pooling' parameter
        if frozen:
            for layer in vgg16.layers[:-1]:
                layer.trainable = False

        return vgg16
    
    def _resnet18_convnet(self, input_shape):
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

        input = keras.layers.Input(shape=input_shape)

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

        output = keras.layers.GlobalAveragePooling2D()(x)
        
        resnet18 = keras.Model(input, output, name="resnet18_convnet")

        return resnet18

    def _cnn128_convnet(self, input_shape):
        def _conv_block(x, filters, kernel_size=3, max_pooling=False):
            x = keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), kernel_initializer='he_uniform')(x)
            x = self.__activation()(x)
            x = keras.layers.BatchNormalization()(x)
            if max_pooling:
                x = keras.layers.MaxPooling2D()(x)

            return x

        input = keras.layers.Input(shape=input_shape)
        
        x = _conv_block(input, 32)
        x = _conv_block(x, 32, True)
        x = _conv_block(x, 64)
        x = _conv_block(x, 64, True)
        x = _conv_block(x, 128)
        x = _conv_block(x, 128, True)
        x = _conv_block(x, 128)
        x = _conv_block(x, 128, True)
        x = _conv_block(x, 128, kernel_size=4)
        
        x = keras.layers.Flatten()(x)
        output = keras.layers.Dense(96)(x)
        
        cnn128 = keras.Model(input, output, name="cnn128_convnet")

        return cnn128

    def _obd_densenset(self, x, hidden_size):
        # local settings
        hidden_size_per_unit = np.round(hidden_size / (self.num_classes - 1)).astype(int)
        
        # define layers
        dense_hidden = [keras.layers.Dense(hidden_size_per_unit) for _ in range(self.num_classes - 1)]
        dense_dropout = [keras.layers.Dropout(self.dropout) for _ in range(self.num_classes - 1)]
        dense_output = [keras.layers.Dense(1) for _ in range(self.num_classes - 1)]
        dense_bn = keras.layers.BatchNormalization()
        dense_lrelu = keras.layers.LeakyReLU()
        dense_sigmoid = keras.layers.Activation('sigmoid')

        xs = [drop(dense_lrelu(hidden(x))) for hidden, drop in zip(dense_hidden, dense_dropout)]
        xs = [dense_sigmoid(dense_bn(output(xc)))[:, 0] for output, xc in zip(dense_output, xs)]
        
        # alt: without batch norm on last FC + Sigmoid
        #xs = [dense_sigmoid(output(xc))[:, 0] for output, xc in zip(dense_output, xs)]
        
        out = tf.concat([tf.expand_dims(xc, axis=1) for xc in xs], axis=1)
        
        # alt: keras model instead of output
        #obd_densenet = keras.Model(x, out, name="obd_densenet")

        return out
    
    def _get_backbone_model(self):
        # get the backbone requested
        backbone = self.nn_backbone

        # get the model based on the backbone
        if backbone == 'resnet18':
            conv_net = self._resnet18_convnet(self.input_shape)
        elif backbone == 'vgg16':
            conv_net = self._vgg16_convnet(self.input_shape, frozen=False)
        elif backbone == 'vgg16_imagenet':
            conv_net = self._vgg16_convnet(self.input_shape, 'imagenet')
        else:
            conv_net = self._cnn128_convnet(self.input_shape)

        return conv_net

    def obd(self):        
        # gather the backbone model based on the network parameters
        conv_net = self._get_backbone_model()

        # set the OBD dense layers after the convolutional network
        obd_dense = self._obd_densenset(conv_net.output, self.obd_hidden_size)
        
        # build the keras neural network model
        model = keras.models.Model(conv_net.input, obd_dense, name="obd")
        
        return model
    
    def clm(self):
        # gather the backbone model based on the network parameters
        conv_net = self._get_backbone_model()
        x = conv_net.output
        
        # apply the dropout layer if requested
        if self.dropout > 0:
            x = keras.layers.Dropout(rate=self.dropout)(x)

        # set up the Cumulative Link Model with the network parameters
        x = keras.layers.Dense(1)(x)
        x = keras.layers.BatchNormalization()(x)
        x = CumulativeLinkModel(self.num_classes, self.clm_link, use_tau=self.use_tau)(x)

        # build the keras neural network model
        model = keras.models.Model(conv_net.input, x, name="clm")
        
        return model

    def resnet18(self):
        # gather the resnet18 backbone
        resnet18 = self._resnet18_convnet(self.input_shape)
        x = resnet18.output
        
        # flatten the convolutional part
        x = keras.layers.Flatten()(x)

        # add the final layers
        x = self.__nominal_final_activation(x)

        # build the keras neural network model
        model = keras.models.Model(resnet18.input, x, name="resnet18")

        return model

    def cnn128(self):
        cnn128 = self._cnn128_convnet(self.input_shape)
        x = cnn128.output

        # add the final layers
        x = self.__nominal_final_activation(x)

        model = keras.models.Model(cnn128.input, x, name="cnn128")
        
        return model
        
    def vgg16(self):
        vgg16 = self._vgg16_convnet(self.input_shape)
        x = vgg16.output

        # add the final layers
        x = self.__nominal_final_activation(x)

        model = keras.models.Model(vgg16.input, x, name="vgg16")
        
        return model
