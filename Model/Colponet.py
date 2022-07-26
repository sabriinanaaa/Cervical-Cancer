# import the necessary packages
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model


class DenseLayer:

    @staticmethod
    def dense_block2(x, blocks, filters):
        for i in range(blocks):
            x = DenseLayer.conv_block2(x, filters)
        return x

    @staticmethod
    def transition_block2(x, reduction):
        chanDim = 3 if K.image_data_format() == 'channels_last' else 1
        x = BatchNormalization(axis=chanDim, epsilon=1.001e-5)(x)
        x = Conv2D(int(K.int_shape(x)[chanDim] * reduction), (1, 1),
                   use_bias=False)(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        x = AveragePooling2D((2, 2), strides=2)(x)
        return x

    @staticmethod
    def conv_block2(x, filters):
        chanDim = 3 if K.image_data_format() == 'channels_last' else 1
        x1 = BatchNormalization(axis=chanDim, epsilon=1.001e-5)(x)
        x1 = Conv2D(4 * filters, (1, 1),
                    use_bias=False)(x1)
        x1 = Activation('relu')(x1)
        x1 = BatchNormalization(axis=chanDim, epsilon=1.001e-5)(x1)
        x1 = Conv2D(filters, (3, 3),
                    padding='same',
                    use_bias=False)(x1)
        x1 = Activation('relu')(x1)
        x = Concatenate(axis=chanDim)([x, x1])
        return x


class ColpoNet:
    @staticmethod
    def build(width, height, depth, filters, classes, reg):

        input_shape = (height, width, depth)
        print(input_shape)

        chanDim = -1
        if K.image_data_format() == "channels_first":
            input_shape = (depth, height, width)
            chanDim = 1
        print(chanDim)

        inputs = Input(shape=input_shape)

        x = Conv2D(filters, (3, 3), input_shape=input_shape,
                   padding='same')(inputs)

        conv1 = Conv2D(filters, (3, 3), padding='same')(x)
        conv2 = Conv2D(filters*4, (3, 3), padding='same')(x)
        x = concatenate([conv1, conv2])

        x = DenseLayer.dense_block2(x, 6, filters)
        x = DenseLayer.transition_block2(x, 0.5)
        x = DenseLayer.dense_block2(x, 12, filters)
        x = DenseLayer.transition_block2(x, 0.5)
        x = DenseLayer.dense_block2(x, 24, filters)
        x = DenseLayer.transition_block2(x, 0.5)
        x = DenseLayer.dense_block2(x, 16, filters)

        x = BatchNormalization(axis=chanDim, epsilon=1.001e-5)(x)

        conv3 = Conv2D(filters * 5, (3, 3), padding='same')(x)
        conv4 = Conv2D(filters * 10, (3, 3), padding='same')(x)
        x = concatenate([conv3, conv4])

        x = Dropout(0.5)(x)
        x = Conv2D(filters*15, (3, 3), use_bias=False,
                   kernel_regularizer=l2(reg))(x)
        x = Dropout(0.5)(x)
        x = Activation("relu")(x)
        x = AveragePooling2D((2, 2), strides=3)(x)
        x = Dropout(0.5)(x)

        x = Flatten()(x)
        x = Dense(classes, kernel_regularizer=l2(reg))(x)
        x = Activation("softmax")(x)

        model = Model(inputs, x)

        return model

