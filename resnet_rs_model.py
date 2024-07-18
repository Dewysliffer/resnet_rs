"""Architecture code for Resnet RS models."""
from typing import Callable, Dict, List, Union

import tensorflow as tf
from absl import logging
from packaging import version
from cbam import CBAMBlock
# Keras has been moved to separate repository in 2.9
if version.parse(tf.__version__) < version.parse("2.8"):
    from tensorflow.python.keras.applications import imagenet_utils
else:
    from keras.applications import imagenet_utils

# tensorflow.python.keras is removed in 2.12
if version.parse(tf.__version__) < version.parse("2.12"):
    from tensorflow.python.keras.utils import layer_utils
else:
    from keras.utils import layer_utils

from tensorflow.python.lib.io import file_io

from block_args import BLOCK_ARGS
from model_utils import (
    allow_bigger_recursion,
    fixed_padding,
    get_survival_probability,
)

BASE_WEIGHTS_URL = (
    "https://github.com/sebastian-sz/resnet-rs-keras/releases/download/v1.0/"
)

WEIGHT_HASHES = {
    "resnet-rs-101-i160.h5": "544b3434d00efc199d66e9058c7f3379",
    "resnet-rs-101-i160_notop.h5": "82d5b90c5ce9d710da639d6216d0f979",
    "resnet-rs-101-i192.h5": "eb285be29ab42cf4835ff20a5e3b5d23",
    "resnet-rs-101-i192_notop.h5": "f9a0f6b85faa9c3db2b6e233c4eebb5b",
    "resnet-rs-50-i160.h5": "69d9d925319f00a8bdd4af23c04e4102",
    "resnet-rs-50-i160_notop.h5": "90daa68cd26c95aa6c5d25451e095529",
}

DEPTH_TO_WEIGHT_VARIANTS = {
    50: [160],
    101: [160, 192],
}


def Conv2DFixedPadding(filters, kernel_size, strides, name):

    def apply(inputs):
        if strides > 1:
            inputs = fixed_padding(inputs, kernel_size)
        return tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="same" if strides == 1 else "valid",
            use_bias=False,
            kernel_initializer=tf.keras.initializers.VarianceScaling(),
            name=name,
        )(inputs)

    return apply


def STEM(bn_epsilon: float, activation: str):
    """ResNet-D type STEM block."""

    def apply(inputs):

        # First stem block
        x = Conv2DFixedPadding(
            filters=32, kernel_size=3, strides=2, name="stem_conv_1"
        )(inputs)
        x = tf.keras.layers.LayerNormalization(
            epsilon=bn_epsilon,
            name="stem_batch_norm_1",
        )(x)
        x = tf.keras.layers.Activation(activation, name="stem_act_1")(x)

        # Second stem block
        x = Conv2DFixedPadding(
            filters=32, kernel_size=3, strides=1, name="stem_conv_2"
        )(x)
        x = tf.keras.layers.LayerNormalization(
            epsilon=bn_epsilon,
            name="stem_batch_norm_2",
        )(x)
        x = tf.keras.layers.Activation(activation, name="stem_act_2")(x)

        # Final Stem block:
        x = Conv2DFixedPadding(
            filters=64, kernel_size=3, strides=1, name="stem_conv_3"
        )(x)
        x = tf.keras.layers.LayerNormalization(
            epsilon=bn_epsilon,
            name="stem_batch_norm_3",
        )(x)
        x = tf.keras.layers.Activation(activation, name="stem_act_3")(x)

        # Replace stem max pool:
        x = Conv2DFixedPadding(
            filters=64, kernel_size=3, strides=2, name="stem_conv_4"
        )(x)
        x = tf.keras.layers.LayerNormalization(
            epsilon=bn_epsilon,
            name="stem_batch_norm_4",
        )(x)
        x = tf.keras.layers.Activation(activation, name="stem_act_4")(x)
        return x

    return apply



def BottleneckBlock(
    filters: int,
    strides: int,
    use_projection: bool,
    bn_epsilon: float,
    activation: str,
    se_ratio: float,
    survival_probability: float,
    name: str = "",
):

    def apply(inputs):
        # bn_axis = 3 if tf.keras.backend.image_data_format() == "channels_last" else 1

        shortcut = inputs

        if use_projection:
            filters_out = filters * 4
            if strides == 2:
                shortcut = tf.keras.layers.AveragePooling2D(
                    pool_size=(2, 2),
                    strides=(2, 2),
                    padding="same",
                    name=name + "projection_pooling",
                )(inputs)
                shortcut = Conv2DFixedPadding(
                    filters=filters_out,
                    kernel_size=1,
                    strides=1,
                    name=name + "projection_conv",
                )(shortcut)
            else:
                shortcut = Conv2DFixedPadding(
                    filters=filters_out,
                    kernel_size=1,
                    strides=strides,
                    name=name + "projection_conv",
                )(inputs)

            shortcut = tf.keras.layers.LayerNormalization(
                epsilon=bn_epsilon,
                name=name + "projection_batch_norm",
            )(shortcut)

        # conv1 layer:
        x = Conv2DFixedPadding(
            filters=filters, kernel_size=1, strides=1, name=name + "conv_1"
        )(inputs)
        x = tf.keras.layers.LayerNormalization(
            epsilon=bn_epsilon,
            name=name + "layer_norm_1",
        )(x)
        x = tf.keras.layers.Activation(activation, name=name + "act_1")(x)

        # conv2 layer:
        x = Conv2DFixedPadding(
            filters=filters, kernel_size=3, strides=strides, name=name + "conv_2"
        )(x)
        x = tf.keras.layers.LayerNormalization(
            epsilon=bn_epsilon,
            name=name + "layer_norm_2",
        )(x)
        x = tf.keras.layers.Activation(activation, name=name + "act_2")(x)

        # conv3 layer:
        x = Conv2DFixedPadding(
            filters=filters * 4, kernel_size=1, strides=1, name=name + "conv_3"
        )(x)
        x = tf.keras.layers.LayerNormalization(
            epsilon=bn_epsilon,
            name=name + "layer_norm_3",
        )(x)

        # 引入注意力机制
        if 0 < se_ratio < 1:
            # x = SE(filters, se_ratio=se_ratio, name=name)(x)
            x = CBAMBlock(reduction_ratio= se_ratio)(x)
        
        # Drop connect
        if survival_probability:
            x = tf.keras.layers.Dropout(
                survival_probability, noise_shape=(None, 1, 1, 1), name=name + "drop"
            )(x)

        x = tf.keras.layers.Add()([x, shortcut])

        return tf.keras.layers.Activation(activation, name=name + "output_act")(x)

    return apply


def BlockGroup(
    filters,
    strides,
    se_ratio,
    bn_epsilon,
    num_repeats,
    activation,
    survival_probability: float,
    name: str,
):
    """Create one group of blocks for the ResNet model."""

    def apply(inputs):
        # Only the first block per block_group uses projection shortcut and strides.
        x = BottleneckBlock(
            filters=filters,
            strides=strides,
            use_projection=True,
            se_ratio=se_ratio,
            bn_epsilon=bn_epsilon,
            activation=activation,
            survival_probability=survival_probability,
            name=name + "block_0_",
        )(inputs)

        for i in range(1, num_repeats):
            x = BottleneckBlock(
                filters=filters,
                strides=1,
                use_projection=False,
                se_ratio=se_ratio,
                activation=activation,
                bn_epsilon=bn_epsilon,
                survival_probability=survival_probability,
                name=name + f"block_{i}_",
            )(x)
        return x

    return apply


def ResNetRS(
    depth: int,
    input_shape=(None, None, 3),
    bn_epsilon=1e-5,
    activation: str = "relu",
    se_ratio=0.25,
    dropout_rate=0.25,
    drop_connect_rate=0.2,
    include_top=True,
    block_args: List[Dict[str, int]] = None,
    model_name="resnet-rs",
    pooling=None,
    weights="imagenet",
    input_tensor=None,
    classes=1000,
    classifier_activation: Union[str, Callable] = "softmax",
):
    """根据提供的参数, 构建Resnet RS模型。

    Parameters:
        :param depth: ResNet网络的深度 即网络中的层数
        :param dropout_rate: 丢弃率 在训练过程中随机丢弃一部分神经元
        :param bn_momentum: 批量归一化 (Batch Normalization) 层中的动量参数
        :param bn_epsilon: 批量归一化层中的一个小正数 Epsilon 防止除以零的错误
        :param activation: 激活函数
        :param block_args: 每个字典包含了构建ResNet块 (如残差块) 所需的参数
        :param se_ratio: Squeeze-and-Excitation(SE)块参数, 控制了SE块中压缩操作的比例, 即如何减少特征图的通道数以进行压缩。
        :param model_name: 模型名称
        :param drop_connect_rate: 在ResNet的残差连接中, drop_connect_rate参数指定了dropout率
        :param include_top: 是否在网络的顶部包含全连接层
        :param weights: None表示使用随机初始化来初始化模型的权重, ImageNet表示加载在ImageNet数据集上预训练的权重,
          默认情况下，如果没有指定具体的变体, 通常会下载最高输入形状(即最大的分辨率)的权重
        :param input_tensor: 指定一个可选的Keras张量(通常是layers.Input()的输出)作为模型的图像输入。
        :param input_shape: 指定模型的输入形状, 比如(224, 224, 3)是一个有效的输入形状
        :param  pooling: 当include_top参数为False时, 指定特征提取的池化模式
            - `None` 表示模型的输出将是最后一个卷积层的4D张量输出
            - `avg` 表示将对最后一个卷积层的输出应用全局平均池化, 这种池化方式有助于减少参数数量，同时保留空间信息。
            - `max` 表示将对最后一个卷积层的输出应用全局最大池化, 与全局平均池化类似
        :param classes: 指定要分类的图像类别数 
        :param classifier_activation: 指定顶部(分类器)层的激活函数。

    Returns:
        方法的返回值是一个tf.keras.Model的实例

    Raises:
        ValueError: 如果指定的weights或input_shape无效则抛出异常
        ValueError: 当使用预训练的顶部层(即模型顶部的全连接层,通常用于分类任务)时,如果classifier_activation参数不是softmax或None, 则会抛出异常。
            因为预训练的分类器层通常被训练为使用softmax激活函数来输出类别概率,如果改为使用其他激活函数(如sigmoid、relu等)则可能导致输出与预训练权重不匹配, 从而影响模型的性能
    """
    # Validate parameters
    available_weight_variants = DEPTH_TO_WEIGHT_VARIANTS[depth]
    if weights == "imagenet":
        max_input_shape = max(available_weight_variants)
        logging.warning(
            f"Received `imagenet` argument without "
            f"explicit weights input size. Picking weights trained with "
            f"biggest available shape: imagenet-i{max_input_shape}"
        )
        weights = f"{weights}-i{max_input_shape}"

    weights_allow_list = [f"imagenet-i{x}" for x in available_weight_variants]
    if not (weights in {*weights_allow_list, None} or file_io.file_exists_v2(weights)):
        raise ValueError(
            "The `weights` argument should be either "
            "`None` (random initialization), `'imagenet'` "
            "(pre-training on ImageNet, with highest available input shape),"
            " or the path to the weights file to be loaded. "
            f"For ResNetRS{depth} the following weight variants are "
            f"available {weights_allow_list} (default=highest)."
            f" Received weights={weights}"
        )

    if weights in weights_allow_list and include_top and classes != 1000:
        raise ValueError(
            f"If using `weights` as `'imagenet'` or any of {weights_allow_list} with "
            f"`include_top` as true, `classes` should be 1000. "
            f"Received classes={classes}"
        )

    # Define input tensor
    if input_tensor is None:
        img_input = tf.keras.layers.Input(shape=input_shape)
    else:
        if not tf.keras.backend.is_keras_tensor(input_tensor):
            img_input = tf.keras.layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # Build stem
    x = STEM(bn_epsilon=bn_epsilon, activation=activation)(
        img_input
    )

    # Build blocks
    if block_args is None:
        block_args = BLOCK_ARGS[depth]

    for i, args in enumerate(block_args):
        survival_probability = get_survival_probability(
            init_rate=drop_connect_rate,
            block_num=i + 2,
            total_blocks=len(block_args) + 1,
        )

        x = BlockGroup(
            filters=args["input_filters"],
            activation=activation,
            strides=(1 if i == 0 else 2),
            num_repeats=args["num_repeats"],
            se_ratio=se_ratio,
            bn_epsilon=bn_epsilon,
            survival_probability=survival_probability,
            name=f"c{i + 2}_",
        )(x)

    # Build head:
    if include_top:
        x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
        if dropout_rate > 0:
            x = tf.keras.layers.Dropout(dropout_rate, name="top_dropout")(x)

        imagenet_utils.validate_activation(classifier_activation, weights)
        x = tf.keras.layers.Dense(
            classes, activation=classifier_activation, name="predictions"
        )(x)
    else:
        if pooling == "avg":
            x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
        elif pooling == "max":
            x = tf.keras.layers.GlobalMaxPooling2D(name="max_pool")(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = tf.keras.Model(inputs, x, name=model_name)

    # Download weights
    if weights in weights_allow_list:
        weights_input_shape = weights.split("-")[-1]  # e. g. "i160"
        weights_name = f"{model_name}-{weights_input_shape}"
        if not include_top:
            weights_name += "_notop"

        filename = f"{weights_name}.h5"
        download_url = BASE_WEIGHTS_URL + filename
        weights_path = tf.keras.utils.get_file(
            fname=filename,
            origin=download_url,
            cache_subdir="models",
            file_hash=WEIGHT_HASHES[filename],
        )
        model.load_weights(weights_path)

    elif weights is not None:
        model.load_weights(weights)

    return model


def ResNetRS50(
    include_top=True,
    weights="imagenet",
    classes=1000,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
):
    """Build ResNet-RS50 model."""
    return ResNetRS(
        depth=50,
        include_top=include_top,
        drop_connect_rate=0.0,
        dropout_rate=0.25,
        weights=weights,
        classes=classes,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classifier_activation=classifier_activation,
        model_name="resnet-rs-50",
    )


def ResNetRS101(
    include_top=True,
    weights="imagenet",
    classes=1000,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
):
    """Build ResNet-RS101 model."""
    return ResNetRS(
        depth=101,
        include_top=include_top,
        drop_connect_rate=0.0,
        dropout_rate=0.25,
        weights=weights,
        classes=classes,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classifier_activation=classifier_activation,
        model_name="resnet-rs-101",
    )


