import tensorflow as tf


def ResBlock_1(X_input, out_channels, scope, flt_size=3, strides=(1,1), bn_mean=0, bn_variance=1):
    """
    每层block的第一个单元
    :param X_input:
    :param out_channels: 输出的channel
    :param scope: 作用域
    :param flt_size: filter_size
    :param strides:
    :param bn_mean: batch_norm 的参数
    :param bn_variance:
    :return:
    """
    size, _, in_channels = tf.shape(input)
    filter1 = [flt_size, flt_size, in_channels, out_channels]
    filter2 = [flt_size, flt_size, out_channels, out_channels]

    with tf.name_scope(scope):
        X_shortcut = X_input

        X1 = tf.nn.conv2d(X_input, filter=filter1, strides=strides, padding='VALID')
        X1 = tf.nn.batch_normalization(X1, mean=bn_mean, variance=bn_variance)
        X1 = tf.nn.relu(X1)

        X2 = tf.nn.conv2d(X1, filter=filter2, strides=(1, 1), padding='SAME')
        X2 = tf.nn.batch_normalization(X2, mean=bn_mean, variance=bn_variance)

        X_output = tf.nn.relu(X2 + X_shortcut)
    return X_output


def ResBlock_2(X_input, out_channels, scope, flt_size=3, strides=(1, 1), bn_mean=0, bn_variance=1):
    """
    每层block的第二个单元
    :param X_input:
    :param out_channels:
    :param scope:
    :param flt_size:
    :param strides:
    :param bn_mean:
    :param bn_variance:
    :return:
    """
    size, _, in_channels = tf.shape(input)
    filter1 = [flt_size, flt_size, in_channels, out_channels]
    filter2 = [flt_size, flt_size, out_channels, out_channels]

    with tf.name_scope(scope):
        X_shortcut = X_input

        X1 = tf.nn.conv2d(X_input, filter=filter1, strides=strides, padding='SAME')
        X1 = tf.nn.batch_normalization(X1, mean=bn_mean, variance=bn_variance)
        X1 = tf.nn.relu(X1)

        X2 = tf.nn.conv2d(X1, filter=filter2, strides=(1, 1), padding='SAME')
        X2 = tf.nn.batch_normalization(X2, mean=bn_mean, variance=bn_variance)

        X_output = tf.nn.relu(X2 + X_shortcut)
    return X_output

def ResNet18():
    """
    注意！！ 这里的bug还没有调好，不能运行
    :return:
    """
    # img = tf.read_file('../good_data/image_a/0a82cc73bfacf284a758e91940a9dea3_a_crop_flip_hori.jpg')
    with tf.variable_scope('ResNet18'):
        inputs = tf.placeholder(dtype=tf.float32, shape=[None, 100, 100, 3], name='input')
        net = tf.nn.conv2d(inputs, filter=[3, 3, 1, 64], strides=(1, 1),padding='VALID', name='Conv1')
        print('SHAPE:\t', tf.shape(net))
        net = tf.nn.max_pool(net, [3, 3], strides=2, name='Pool1')
        with tf.name_scope('Block1'):
            net = ResBlock_1(net, 64, 'unit1', 3, (2, 2))
            net = ResBlock_2(net, 64, 'unit2')
        with tf.name_scope('Block2'):
            net = ResBlock_1(net, 128, 'unit1', 3, (2, 2))
            net = ResBlock_2(net, 128, 'unit2')
        with tf.name_scope('Block3'):
            net = ResBlock_1(net, 256, 'unit1', 3, (2, 2))
            net = ResBlock_2(net, 256, 'unit2')
        with tf.name_scope('Block4'):
            net = ResBlock_1(net, 512, 'unit1', 3, (2, 2))
            net = ResBlock_2(net, 512, 'unit2')

        net = tf.reduce_mean(net, [1, 2], name='global_pool', keep_dims=True)
        print(net.shape)
        net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
        print(net.shape)
        W = tf.get_variable(dtype=tf.float32, shape=[net.shape[0], 1], name='W')
        b = tf.get_variable(dtype=tf.float32, shape=[None, 1], name='b')
        net = tf.add(tf.matmul(W, net), b)

    return net


if __name__ == "__main__":
    ResNet18()