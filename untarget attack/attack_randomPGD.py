# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.stats as st
from PIL import Image
from scipy.misc import imread
from scipy.misc import imresize
from tensorflow.contrib.slim.nets import resnet_v1, inception, vgg
import foolbox
from foolbox.batching import run_parallel_attack

import time
begin_time = time.time()

os.environ['CUDA_VISIBLE_DEVICES']='3'

slim = tf.contrib.slim

tf.flags.DEFINE_string(
    'checkpoint_path', '', 'Path to checkpoint for inception network.')
tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')
tf.flags.DEFINE_string(
    'output_dir', '', 'Output directory with images.')

FLAGS = tf.flags.FLAGS

# 声明一些攻击参vi
CHECKPOINTS_DIR = './model/'
#CHECKPOINTS_DIR = FLAGS.checkpoint_path
model_checkpoint_map = {
    'inception_v1': os.path.join(CHECKPOINTS_DIR, 'inception_v1', 'inception_v1.ckpt'),
    'resnet_v1_50': os.path.join(CHECKPOINTS_DIR, 'resnet_v1_50', 'model.ckpt-49800'),
    'vgg_16': os.path.join(CHECKPOINTS_DIR, 'vgg_16', 'vgg_16.ckpt')}

#input_dir = './dev_data/'
#output = './output/'

batch_size = 22

# 在图片数据输入模型前，做一些预处理
def preprocess_for_model(images, model_type):
    if 'inception' in model_type.lower():
        images = tf.image.resize_bilinear(images, [224, 224], align_corners=False)
        # tensor-scalar operation
        images = (images / 255.0) * 2.0 - 1.0
        return images

    if 'resnet' in model_type.lower() or 'vgg' in model_type.lower():
        _R_MEAN = 123.68
        _G_MEAN = 116.78
        _B_MEAN = 103.94
        images = tf.image.resize_bilinear(images, [224, 224], align_corners=False)
        tmp_0 = images[:, :, :, 0] - _R_MEAN
        tmp_1 = images[:, :, :, 1] - _G_MEAN
        tmp_2 = images[:, :, :, 2] - _B_MEAN
        images = tf.stack([tmp_0, tmp_1, tmp_2], 3)
        return images


def load_images_with_target_label(input_dir):
    images = []
    filenames = []
    target_labels = []
    idx = 0

    dev = pd.read_csv(os.path.join(input_dir, 'dev.csv'))
    filename2label = {dev.iloc[i]['filename']: dev.iloc[i]['targetedLabel'] for i in range(len(dev))}
    for filename in filename2label.keys():
        image = imread(os.path.join(input_dir, filename), mode='RGB')
        image = imresize(image, [224, 224])
        image = (image / 255.0)
        images.append(image)
        filenames.append(filename)
        target_labels.append(filename2label[filename])
        idx += 1
        if idx == batch_size:
            images = np.array(images)
            yield filenames, images, target_labels
            filenames = []
            images = []
            target_labels = []
            idx = 0
    if idx > 0:
        images = np.array(images)
        yield filenames, images, target_labels


def save_images(images, filenames, output_dir):
    for i, filename in enumerate(filenames):
        image = (((images[i] + 1.0) * 0.5) * 255.0).astype(np.uint8)
        # resize back to [299, 299]
        image = imresize(image, [299, 299])
        Image.fromarray(image).save(os.path.join(output_dir, filename), format='PNG')


def check_or_create_dir(directory):
    """Check if directory exists otherwise create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def ens_model(x):

    #input x remains in[0,1]
    print('image.shape:', x.shape)
    image = (x * 255.0)
    num_classes = 110
    processed_incv1 = preprocess_for_model(image, 'inception_v1')
    with slim.arg_scope(inception.inception_v1_arg_scope()):
        logits_inc_v1, end_points_inc_v1 = inception.inception_v1(
            processed_incv1, num_classes=num_classes, is_training=False, scope='InceptionV1')

    processed_imgs_res_v1_50 = preprocess_for_model(image, 'resnet_v1_50')
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        logits_res_v1_50, end_points_res_v1_50 = resnet_v1.resnet_v1_50(
            processed_imgs_res_v1_50, num_classes=num_classes, is_training=False, scope='resnet_v1_50')

    end_points_res_v1_50['logits'] = tf.squeeze(end_points_res_v1_50['resnet_v1_50/logits'], [1, 2])
    end_points_res_v1_50['probs'] = tf.nn.softmax(end_points_res_v1_50['logits'])

    # image = (((x + 1.0) * 0.5) * 255.0)#.astype(np.uint8)
    processed_imgs_vgg_16 = preprocess_for_model(image, 'vgg_16')
    with slim.arg_scope(vgg.vgg_arg_scope()):
        logits_vgg_16, end_points_vgg_16 = vgg.vgg_16(
            processed_imgs_vgg_16, num_classes=num_classes, is_training=False, scope='vgg_16')

    end_points_vgg_16['logits'] = end_points_vgg_16['vgg_16/fc8']
    end_points_vgg_16['probs'] = tf.nn.softmax(end_points_vgg_16['logits'])

    ########################
    #one_hot = tf.one_hot(y, num_classes)
    ########################

    logits = (end_points_inc_v1['Logits'] + end_points_res_v1_50['logits'] + end_points_vgg_16['logits']) / 3.0
    print('logits.shape:', logits.shape)

    return logits

def attack_randomPGD(input_dir, output_dir):

    batch_shape = [batch_size, 224, 224, 3]
    check_or_create_dir(output_dir)

    with tf.Graph().as_default():
        # Prepare graph
        # preprocessing for model input,
        # note that images for all classifier will be normalized to be in [0, 1]
        images = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
        label = tf.constant(np.zeros([batch_size]), tf.int64)

        logits = ens_model(images)

        # Run computation
        s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV1'))
        s2 = tf.train.Saver(slim.get_model_variables(scope='resnet_v1_50'))
        s3 = tf.train.Saver(slim.get_model_variables(scope='vgg_16'))

        with tf.Session() as sess:
            s1.restore(sess, model_checkpoint_map['inception_v1'])
            s2.restore(sess, model_checkpoint_map['resnet_v1_50'])
            s3.restore(sess, model_checkpoint_map['vgg_16'])
            fmodel = foolbox.models.TensorFlowModel(images, logits, (0,1))
            attack_create_fn = foolbox.attacks.RandomPGD
            criterion = foolbox.criteria.Misclassification()


            for filenames, raw_images, target_labels in load_images_with_target_label(input_dir):
                #processed_imgs_ = sess.run(processed_imgs, feed_dict={raw_inputs: raw_images})
                advs = run_parallel_attack(attack_create_fn, fmodel, criterion, raw_images, target_labels)
                adv_images = np.array(advs)
                print(adv_images.shape)
                save_images(adv_images, filenames, output_dir)




if __name__ == '__main__':
    input_dir = 'dev_data'
    output_dir = 'output_PGD'
    #input_dir = FLAGS.input_dir
    #output_dir = FLAGS.output_dir
    attack_randomPGD(input_dir, output_dir)
    print('Time cost is %f s' % (time.time() - begin_time))
    pass