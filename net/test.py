import tensorflow as tf
from net import load_data
import matplotlib.pyplot as plt
import matplotlib.image as Image


def main(_):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.8
    img, label = load_data.get_data('test/test.tfrecords', batch=1)

    ckpt = tf.train.get_checkpoint_state('D:\python文档\FPM/train_dir/')
    # print(ckpt)
    saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
    # saver=tf.train.import_meta_graph('train_dir/U_net_barch1_1000.ckpt.meta')
    graph = tf.get_default_graph()

    xs = graph.get_tensor_by_name('inputs/x_inputs:0')
    ys = graph.get_tensor_by_name('inputs/y_inputs:0')
    prediction = graph.get_tensor_by_name('output/Relu:0')
    loss = graph.get_tensor_by_name('loss_function/Mean:0')

    sess = tf.Session(config=config)
    saver.restore(sess, ckpt.model_checkpoint_path)
    img1 = sess.run(prediction, feed_dict={xs: img, ys: label})

    img1 = img1.reshape([672, 672])
    plt.imshow(img1, cmap='gray')
    plt.show()
    Image.imsave('训练结果/24.png', img1, cmap='gray')
    print(sess.run(loss, feed_dict={xs: img, ys: label}))


if __name__ == '__main__':
    tf.app.run()
