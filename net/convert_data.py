from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import os
import matplotlib.image as Image
import numpy as np


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'dataset_dir',
    None,
    '所有低分辨图像的储存路径')
tf.app.flags.DEFINE_string(
    'label_dir',
    None,
    '所有高分辨图像的储存路径')
tf.app.flags.DEFINE_string(
    'tfrecords_path',
    None,
    'tfrecords文件存放路径')


def get_subdirs(dataset_dir):
    sub_dirs = []
    dir_list = os.listdir(dataset_dir)
    for i in range(len(dir_list)):
        path = os.path.join(dataset_dir, dir_list[i])
        sub_dirs.append(path)
    return sub_dirs


def get_image(data_dir):
    images = os.listdir(data_dir)
    images = sorted(images, key=lambda x: os.path.getmtime(os.path.join(data_dir, x)))
    xs=[]
    for i in range(len(images)):
        image = Image.imread(os.path.join(data_dir, images[i]))
        xs.append(image)
    xs = np.array(xs)
    xs = xs.swapaxes(0, 1)
    xs = xs.swapaxes(1, 2)
    return xs


def get_label(label_path):
    label = Image.imread(label_path)
    return label


def run(dataset_dir, label_dir, tfrecords_path):
    # dataset_dir='data/Ideal'所有低分辨图像的储存路径
    # label_dir='data/label' 所有高分辨图形的储存路径
    data_dirs = get_subdirs(dataset_dir)
    label_paths = get_subdirs(label_dir)

    tfrecords = tfrecords_path
    writer = tf.python_io.TFRecordWriter(tfrecords)
    for i in range(len(data_dirs)):
        img_raw = get_image(data_dirs[i])
        img_raw = img_raw.tostring()
        label = get_label(label_paths[i])
        label = label.tostring()

        example = tf.train.Example(features=tf.train.Features(
            feature={
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label]))
            }))
        writer.write(example.SerializeToString())
    writer.close()


if __name__ == '__main__':
    run('data/672/', 'label/672/24' 'data/train.tfrecords')
