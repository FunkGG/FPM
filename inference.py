import os
import sys
import tensorflow as tf
import net.convert_data as convert_data

sys.path.append('net/')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'dataset_name',
    '672',
    '所有低分辨图像的储文件夹')
tf.app.flags.DEFINE_string(
    'label_name',
    'label/672',
    '所有高分辨图像的储存文件夹')
# dataset_dir = 'data/'+FLAGS.dataset_name+'/'
# label_dir = 'data/'+FLAGS.label_name+'/'
# tfrecords_path = 'data/train.tfrecords'
dataset_dir = 'test/'+FLAGS.dataset_name+'/'
label_dir = 'test/'+FLAGS.label_name+'/'
tfrecords_path = 'test/test.tfrecords'


def convert_tfrecord():
    convert_data.run(dataset_dir, label_dir, tfrecords_path)


def train_net():
    os.system('python net/train.py')


def test_net():
    os.system('python net/test.py')


def main(_):
    # convert_tfrecord()#成功
    # train_net()#0.0002780414
    test_net()#[0.04483341]


if __name__ == '__main__':
    tf.app.run()
