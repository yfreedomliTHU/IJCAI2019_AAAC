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

os.environ['CUDA_VISIBLE_DEVICES']='0'

slim = tf.contrib.slim

tf.flags.DEFINE_string(
    'checkpoint_path', '', 'Path to checkpoint for inception network.')
tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')
tf.flags.DEFINE_string(
    'output_dir', '', 'Output directory with images.')

FLAGS = tf.flags.FLAGS

# 声明一些攻击参vi
#CHECKPOINTS_DIR = './checkpoint/'
CHECKPOINTS_DIR = FLAGS.checkpoint_path
model_checkpoint_map = {
    'inception_v1': os.path.join(CHECKPOINTS_DIR, 'inception_v1', 'inception_v1.ckpt'),
    'resnet_v1_50': os.path.join(CHECKPOINTS_DIR, 'resnet_v1_50', 'model.ckpt-49800'),
    'vgg_16': os.path.join(CHECKPOINTS_DIR, 'vgg_16', 'vgg_16.ckpt')}

#input_dir = './dev_data/'
#output = './output/'

max_epsilon = 16.0
num_iter = 15
batch_size = 11
momentum = 1.0


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


def target_graph(x, y, i, x_max, x_min, grad):
    eps = 2.0 * max_epsilon / 255.0
    alpha = eps / num_iter
    num_classes = 110

    with slim.arg_scope(inception.inception_v1_arg_scope()):
        logits_inc_v1, end_points_inc_v1 = inception.inception_v1(
            x, num_classes=num_classes, is_training=False, scope='InceptionV1')

    # rescale pixle range from [-1, 1] to [0, 255] for resnet_v1 and vgg's input
    image = (((x + 1.0) * 0.5) * 255.0)
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
    one_hot = tf.one_hot(y, num_classes)
    ########################

    logits = (logits_inc_v1 + logits_res_v1_50 + logits_vgg_16) / 3.0
    #auxlogits = end_points_inc_v1['AuxLogits']
    cross_entropy = tf.losses.softmax_cross_entropy(one_hot,
                                                    logits,
                                                    label_smoothing=0.0,
                                                    weights=1.0)
    """
    cross_entropy += tf.losses.softmax_cross_entropy(one_hot,
                                                     auxlogits,
                                                     label_smoothing=0.0,
                                                     weights=0.4)
    """
    noise = tf.gradients(cross_entropy, x)[0]
    noise = noise / tf.reshape(
        tf.contrib.keras.backend.std(tf.reshape(noise, [batch_size, -1]), axis=1),
        [batch_size, 1, 1, 1])
    noise = momentum * grad + noise
    noise = noise / tf.reshape(
        tf.contrib.keras.backend.std(tf.reshape(noise, [batch_size, -1]), axis=1),
        [batch_size, 1, 1, 1])
    x = x - alpha * tf.clip_by_value(tf.round(noise), -2, 2)
    x = tf.clip_by_value(x, x_min, x_max)
    i = tf.add(i, 1)
    return x, y, i, x_max, x_min, noise


def stop(x, y, i, x_max, x_min, grad):
    return tf.less(i, num_iter)


# Momentum Iterative FGSM
def target_mi_fgsm_attack(input_dir, output_dir):
    # some parameter
    eps = 2.0 * max_epsilon / 255.0
    batch_shape = [batch_size, 224, 224, 3]

    check_or_create_dir(output_dir)

    with tf.Graph().as_default():
        # Prepare graph
        raw_inputs = tf.placeholder(tf.uint8, shape=[None, 299, 299, 3])

        # preprocessing for model input,
        # note that images for all classifier will be normalized to be in [-1, 1]
        processed_imgs = preprocess_for_model(raw_inputs, 'inception_v1')

        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        x_max = tf.clip_by_value(x_input + eps, -1.0, 1.0)
        x_min = tf.clip_by_value(x_input - eps, -1.0, 1.0)

        #     y = tf.constant(np.zeros([batch_size]), tf.int64)
        y = tf.placeholder(tf.int32, shape=[batch_size])
        i = tf.constant(0)
        grad = tf.zeros(shape=batch_shape)
        x_adv, _, _, _, _, _ = tf.while_loop(stop, target_graph, [x_input, y, i, x_max, x_min, grad])

        # Run computation
        s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV1'))
        s2 = tf.train.Saver(slim.get_model_variables(scope='resnet_v1_50'))
        s3 = tf.train.Saver(slim.get_model_variables(scope='vgg_16'))

        with tf.Session() as sess:
            s1.restore(sess, model_checkpoint_map['inception_v1'])
            s2.restore(sess, model_checkpoint_map['resnet_v1_50'])
            s3.restore(sess, model_checkpoint_map['vgg_16'])

            for filenames, raw_images, target_labels in load_images_with_target_label(input_dir):
                processed_imgs_ = sess.run(processed_imgs, feed_dict={raw_inputs: raw_images})
                adv_images = sess.run(x_adv, feed_dict={x_input: processed_imgs_, y: target_labels})
                save_images(adv_images, filenames, output_dir)


if __name__ == '__main__':
    #input_dir = 'dev_data'
    #output_dir = 'output_iter_'+str(num_iter)
    input_dir = FLAGS.input_dir
    output_dir = FLAGS.output_dir
    target_mi_fgsm_attack(input_dir, output_dir)
    pass