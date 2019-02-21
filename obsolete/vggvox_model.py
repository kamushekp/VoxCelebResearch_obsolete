import scipy.io as sio
import numpy as np
import keras.backend as K
from keras.layers import Input, GlobalAveragePooling2D, Reshape
from keras.layers.convolutional import Conv2D, ZeroPadding2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Lambda, Activation
from keras.models import Model

import constants as c


# Block of layers: Conv --> BatchNorm --> ReLU --> Pool
def conv_bn_pool(inp_tensor,conv_filters, conv_kernel_size, conv_strides, conv_pad, pool_type = "", pool_size=(2, 2),pool_strides=None):
    x = ZeroPadding2D(padding=conv_pad)(inp_tensor)
    x = Conv2D(filters=conv_filters,kernel_size=conv_kernel_size, strides=conv_strides, padding='valid')(x)
    x = BatchNormalization(epsilon=1e-5,momentum=1)(x)
    x = Activation('relu')(x)
    
    if pool_type == 'max':
        return MaxPooling2D(pool_size=pool_size,strides=pool_strides)(x)
    elif pool_type == 'avg':
        return AveragePooling2D(pool_size=pool_size,strides=pool_strides)(x)
    
    return x


# Block of layers: Conv --> BatchNorm --> ReLU --> Dynamic average pool (fc6 -> apool6 only)
def conv_bn_dynamic_apool(inp_tensor,conv_filters,conv_kernel_size,conv_strides,conv_pad):
    x = ZeroPadding2D(padding=conv_pad)(inp_tensor)
    x = Conv2D(filters=conv_filters,kernel_size=conv_kernel_size, strides=conv_strides, padding='valid')(x)
    x = BatchNormalization(epsilon=1e-5,momentum=1)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1,1,conv_filters))(x)
    return x


def vggvox_model():
    inp = Input(c.INPUT_SHAPE,name='input')
    x = conv_bn_pool(inp,conv_filters=96,conv_kernel_size=(7,7),conv_strides=(2,2),conv_pad=(1,1),
        pool_type='max',pool_size=(3,3),pool_strides=(2,2))
    x = conv_bn_pool(x,conv_filters=256,conv_kernel_size=(5,5),conv_strides=(2,2),conv_pad=(1,1),
        pool_type='max',pool_size=(3,3),pool_strides=(2,2))
    x = conv_bn_pool(x,conv_filters=384,conv_kernel_size=(3,3),conv_strides=(1,1),conv_pad=(1,1))
    x = conv_bn_pool(x,conv_filters=256,conv_kernel_size=(3,3),conv_strides=(1,1),conv_pad=(1,1))
    x = conv_bn_pool(x,conv_filters=256,conv_kernel_size=(3,3),conv_strides=(1,1),conv_pad=(1,1), pool_type='max',pool_size=(5,3),pool_strides=(3,2))
    x = conv_bn_dynamic_apool(x,conv_filters=4096,conv_kernel_size=(9,1),conv_strides=(1,1),conv_pad=(0,0))
    x = conv_bn_pool(x,conv_filters=1024,conv_kernel_size=(1,1),conv_strides=(1,1),conv_pad=(0,0))
    x = Lambda(lambda y: K.l2_normalize(y, axis=3))(x)
    x = Conv2D(filters=1024,kernel_size=(1,1), strides=(1,1), padding='valid')(x)
    m = Model(inp, x)
    return m


def test():
    model = vggvox_model()
    num_layers = len(model.layers)

    x = np.random.randn(1,512,300,1)
    outputs = []

    for i in range(num_layers):
        get_ith_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                      [model.layers[i].output])	
        layer_output = get_ith_layer_output([x, 0])[0] 	# output in test mode = 0
        outputs.append(layer_output)

    for i in range(11):
        print("Shape of layer {} output:{}".format(i, outputs[i].shape))


if __name__ == '__main__':
    test()

