import tensorflow as tf
from net import load_data
import matplotlib.pyplot as plt
import matplotlib.image as Image
from net import U_Net


def loss_func(v_xs, v_ys):
    result = tf.reduce_mean(tf.reduce_mean(tf.square(v_xs-v_ys), axis=[1, 2, 3]))
    return result


def main(_):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.8
    with tf.name_scope('inputs'):
        xs = tf.placeholder(tf.float32, [None, 672, 672, 256], name='x_inputs')
        ys = tf.placeholder(tf.float32, [None, 672, 672, 1], name='y_inputs')
    prediction = U_Net.U_Net(xs)
    with tf.name_scope('loss_function'):
        loss = loss_func(prediction, ys)
        tf.summary.scalar('loss', loss)
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

    img, label = load_data.get_data('data/train.tfrecords', batch=1)  # 读取数据
    sess = tf.Session(config=config)
    # merged = tf.summary.merge_all()
    # writer = tf.summary.FileWriter("logs/", sess.graph)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    for i in range(900):
        sess.run(train_step, feed_dict={xs: img, ys: label})
        if i % 10 == 0:
            print(sess.run(loss, feed_dict={xs: img, ys: label}))
            # result = sess.run(merged, feed_dict={xs: img, ys: label})
            # writer.add_summary(result, i)
    saver.save(sess, "train_dir/U_net_barch1_1000.ckpt")

    img11 = sess.run(prediction, feed_dict={xs:img,ys:label})
    img11 = img11.reshape([672, 672])
    Image.imsave('训练结果/23.png', img11, cmap='gray')
    # plt.imshow(img11,cmap='gray')
    # plt.show()


if __name__ == '__main__':
    tf.app.run()
