import tensorflow as tf
import numpy as np


def get_data(tfrecords,batch):
    filename_queue = tf.train.string_input_producer([tfrecords],)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.string),
                                           'img_raw' : tf.FixedLenFeature([], tf.string)
                                       })

    image = tf.decode_raw(features['img_raw'], tf.float32)
    image = tf.reshape(image, [672,672,256])
    label = tf.decode_raw(features['label'], tf.float32)
    label = tf.reshape(label, [672, 672,1])
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        '''依次读取数据，返回为np.array'''
        image_batch=np.zeros([batch,672,672,256])
        label_batch=np.zeros([batch,672,672,1])
        for i in range(batch):
            image_single,label_single = sess.run([image, label])  # 在会话中取出image和label
            image_batch[i, :, :, :] = image_single
            label_batch[i, :, :, :] = label_single
        '''批量读取数据，返回为tensor'''
        # image_batch,label_batch=tf.train.batch(
        #     [image,label],
        #     batch_size=1,
        #     capacity=4)
        coord.request_stop()
        coord.join(threads)
    return image_batch,


if __name__ == '__main__':
    a, b = get_data('data/train.tfrecords')