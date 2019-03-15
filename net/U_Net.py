import tensorflow as tf



def U_Net(inputs
               # num_classes=1000,
               # is_training=True,
               # dropout_keep_prob=0.5,
               # spatial_squeeze=True,
               # scope='U_Net'
          ):
    slim = tf.contrib.slim
    with slim.arg_scope([slim.conv2d,slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0,0.01),
                        weights_regularizer=slim.l2_regularizer(0.0005)

                        ):
        conv1 = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], padding='SAME', scope='conv1')
        pool1 = slim.max_pool2d(conv1, [2, 2], 2, scope='pool1')
        conv2 = slim.repeat(pool1, 2, slim.conv2d, 128, [3, 3], padding='SAME', scope='conv2')
        pool2 = slim.max_pool2d(conv2, [2, 2], stride=2, scope='pool2')
        conv3 = slim.repeat(pool2, 2, slim.conv2d, 256, [3, 3], padding='SAME', scope='conv3')
        pool3 = slim.max_pool2d(conv3, [2, 2], stride=2, scope='pool3')
        conv4 = slim.repeat(pool3, 2, slim.conv2d, 512, [3, 3], padding='SAME', scope='conv4')
        pool4 = slim.max_pool2d(conv4, [2, 2], stride=2, scope='pool4')
        conv5 = slim.repeat(pool4, 2, slim.conv2d, 1024, [3, 3], padding='SAME', scope='conv5')

        up1 = slim.conv2d_transpose(conv5,512,[2,2],stride=2,padding='SAME')
        up1 = tf.concat([conv4, up1], -1, name='up1')
        conv6 = slim.repeat(up1, 2, slim.conv2d, 512, [3, 3], padding='SAME', scope='conv6')
        up2 = slim.conv2d_transpose(conv6, 256, [2, 2], stride=2, padding='SAME')
        up2 = tf.concat([conv3, up2], -1, name='up2')
        conv7 = slim.repeat(up2, 2, slim.conv2d, 128, [3, 3], padding='SAME', scope='conv7')
        up3 = slim.conv2d_transpose(conv7, 256, [2, 2], stride=2, padding='SAME')
        up3 = tf.concat([conv2, up3], -1, name='up3')
        conv8 = slim.repeat(up3, 2, slim.conv2d, 128, [3, 3], padding='SAME', scope='conv8')
        up4 = slim.conv2d_transpose(conv8, 256, [2, 2], stride=2, padding='SAME')
        up4 = tf.concat([conv1, up4], -1, name='up4')

        conv9 = slim.repeat(up4, 2, slim.conv2d, 64, [3, 3], padding='SAME', scope='conv9')
        net = slim.conv2d(conv9, 1, [3, 3], padding='SAME', scope='output')
        return net
