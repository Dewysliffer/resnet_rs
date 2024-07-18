import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.layers import Layer, Conv2D, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Multiply


class ChannelAttentionModule(Layer):
    def __init__(self, reduction_ratio, **kwargs):
        super(ChannelAttentionModule, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        self.avg_pool = GlobalAveragePooling2D()
        self.max_pool = GlobalMaxPooling2D()
        input_channel = input_shape[-1]
        num_squeeze = input_channel // self.reduction_ratio
        self.fc1 = Dense(num_squeeze, activation=None, kernel_initializer=GlorotUniform(), kernel_regularizer=l2(0.0005))
        self.fc2 = Dense(input_channel, activation=None, kernel_initializer=GlorotUniform(), kernel_regularizer=l2(0.0005))

    def call(self, inputs):
        avg_pool = self.avg_pool(inputs)
        avg_pool = self.fc1(avg_pool)
        avg_pool = self.fc2(avg_pool)
        
        max_pool = self.max_pool(inputs)
        max_pool = self.fc1(max_pool)
        max_pool = self.fc2(max_pool)
        
        scale = tf.nn.sigmoid(avg_pool + max_pool)
        return Multiply()([scale, inputs])
		
class SpatialAttentionModule(Layer):
    def __init__(self, kernel_size, **kwargs):
        super(SpatialAttentionModule, self).__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv1 = Conv2D(1, self.kernel_size, padding='same', activation='sigmoid')

    def call(self, inputs):
        avg_out = tf.reduce_mean(inputs, axis=3, keepdims=True)
        max_out = tf.reduce_max(inputs, axis=3, keepdims=True)
        concat = tf.concat([avg_out, max_out], axis=3)
        scale = self.conv1(concat)
        return scale * inputs
		
		
class CBAMBlock(Layer):
    def __init__(self, reduction_ratio=16, **kwargs):
        super(CBAMBlock, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.channel_attention = ChannelAttentionModule(self.reduction_ratio)
        self.spatial_attention = SpatialAttentionModule(kernel_size=7)

    def call(self, inputs):
        channel_attention = self.channel_attention(inputs)
        spatial_attention = self.spatial_attention(channel_attention)
        return spatial_attention

# use
# cbam_block = CBAMBlock(reduction_ratio=16)