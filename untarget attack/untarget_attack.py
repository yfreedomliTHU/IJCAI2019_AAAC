# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.stats as st
import PIL
from PIL import Image
from scipy.misc import imread
from scipy.misc import imresize
from tensorflow.contrib.slim.nets import resnet_v1, inception, vgg
import time
start = time.clock()
slim = tf.contrib.slim
import time
begin_time = time.time()

os.environ['CUDA_VISIBLE_DEVICES']='5'

tf.flags.DEFINE_string(
    'checkpoint_path', './model', 'Path to checkpoint for inception network.')
tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')
tf.flags.DEFINE_string(
    'output_dir', '', 'Output directory with images.')

FLAGS = tf.flags.FLAGS
CHECKPOINTS_DIR = FLAGS.checkpoint_path
model_checkpoint_map = {
    'inception_v1': os.path.join(CHECKPOINTS_DIR, 'inception_v1', 'inception_v1.ckpt'),
    'resnet_v1_50': os.path.join(CHECKPOINTS_DIR, 'resnet_v1_50', 'model.ckpt-49800'),
    'vgg_16': os.path.join(CHECKPOINTS_DIR, 'vgg_16', 'vgg_16.ckpt')}


max_epsilon = 16.0
num_iter = 15
batch_size = 32
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


# 加载评测图片
def load_images_with_true_label(input_dir):
    file_name = os.listdir(input_dir)

    images = []
    filenames = []
    #true_labels = []
    idx = 0
    count = 0
    #dev = pd.read_csv(os.path.join(input_dir, 'dev.csv'))
    #filename2label = {dev.iloc[i]['filename']: dev.iloc[i]['trueLabel'] for i in range(len(dev))}
    for filename in file_name:
        image = imread(os.path.join(input_dir, filename), mode='RGB')
        #print(image.shape)
        image = imresize(image, [299, 299])
        images.append(image)
        filenames.append(filename)
        #true_labels.append(filename2label[filename])

        idx += 1
        if idx == batch_size:
            count += 1
            print('================================batch'+str(count)+'processed!===================================')
            images = np.array(images)
            yield filenames, images
            filenames = []
            images = []
            #true_labels = []
            idx = 0
    '''
    if idx > 0:
        count +=1
        print('================================batch' + str(count) + 'processed!===================================')
        images = np.array(images)
        print(images.shape)
        yield filenames, images
    '''




def save_images(images, filenames, output_dir):
    for i, filename in enumerate(filenames):
        image = (((images[i] + 1.0) * 0.5) * 255.0).astype(np.uint8)
        #print(image.shape)
        # resize back to [299, 299]
        image = imresize(image, [299, 299])
        Image.fromarray(image).save(os.path.join(output_dir, filename), format='JPEG')


def check_or_create_dir(directory):
    """Check if directory exists otherwise create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)


# 定义MI_FGSM迭代攻击的计算图
def non_target_graph(x, y, i, x_max, x_min, grad):
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
    # Using model predictions as ground truth to avoid label leaking
    pred = tf.argmax(end_points_inc_v1['Predictions'] + end_points_res_v1_50['probs'] + end_points_vgg_16['probs'], 1)
    first_round = tf.cast(tf.equal(i, 0), tf.int64)
    y = first_round * pred + (1 - first_round) * y
    one_hot = tf.one_hot(y, num_classes)
    ########################
    logits = (end_points_inc_v1['Logits'] + end_points_res_v1_50['logits'] + end_points_vgg_16['logits']) / 3.0
    cross_entropy = tf.losses.softmax_cross_entropy(one_hot,
                                                    logits,
                                                    label_smoothing=0.0,
                                                    weights=1.0)
    noise = tf.gradients(cross_entropy, x)[0]
    noise = noise / tf.reduce_mean(tf.abs(noise), [1, 2, 3], keep_dims=True)
    noise = momentum * grad + noise
    x = x + alpha * tf.sign(noise)
    x = tf.clip_by_value(x, x_min, x_max)
    i = tf.add(i, 1)
    return x, y, i, x_max, x_min, noise


def stop(x, y, i, x_max, x_min, grad):
    return tf.less(i, num_iter)
'''
def filter(image):
    image_ = np.array(image)
    _image = np.asarray(Image.fromarray(image_).convert("RGB")).astype(np.float32)
    left = 72
    right = 227
    top = 72
    bottom = 227

    new_image = _image.copy()
    n = 1
    for i in range(72):
        new_image[left:right, top:bottom] += _image[(left - i):(right - i), top:bottom]
        new_image[left:right, top:bottom] += _image[(left + i):(right + i), top:bottom]
        new_image[left:right, top:bottom] += _image[left:right, (top - i):(bottom - i)]
        new_image[left:right, top:bottom] += _image[left:right, (top + i):(bottom + i)]
        n += 4

    new_image[left:right, top:bottom] /= n
    new_image = new_image.astype(np.uint8)
    return new_image
'''

# Momentum Iterative FGSM
def non_target_mi_fgsm_attack(input_dir, output_dir):
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

        y = tf.constant(np.zeros([batch_size]), tf.int64)
        # y = tf.placeholder(tf.int32, shape=[batch_size])
        i = tf.constant(0)
        grad = tf.zeros(shape=batch_shape)
        x_adv, _, _, _, _, _ = tf.while_loop(stop, non_target_graph, [x_input, y, i, x_max, x_min, grad])

        # Run computation
        s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV1'))
        s2 = tf.train.Saver(slim.get_model_variables(scope='resnet_v1_50'))
        s3 = tf.train.Saver(slim.get_model_variables(scope='vgg_16'))

        with tf.Session() as sess:
            s1.restore(sess, model_checkpoint_map['inception_v1'])
            s2.restore(sess, model_checkpoint_map['resnet_v1_50'])
            s3.restore(sess, model_checkpoint_map['vgg_16'])

            for filenames, raw_images in load_images_with_true_label(input_dir):
                processed_imgs_ = sess.run(processed_imgs, feed_dict={raw_inputs: raw_images})
                adv_images = sess.run(x_adv, feed_dict={x_input: processed_imgs_})
                save_images(adv_images, filenames, output_dir)


if __name__ == '__main__':


    '''
    input_dir = FLAGS.input_dir
    output_dir = FLAGS.output_dir
    non_target_mi_fgsm_attack(input_dir, output_dir)
    elapsed = (time.clock() - start)
    print("Run Time:", elapsed)
    pass
    '''
    clean_path = 'IJCAI_2019_AAAC_train_processed'
    adv_path = 'IJCAI_2019_AAAC_train_MIFSGM'
    Class_list = os.listdir(clean_path)
    count = 0
    for Class in Class_list:
        Input_dir = os.path.join(clean_path, Class)
        Output_dir = os.path.join(adv_path, Class)
        non_target_mi_fgsm_attack(Input_dir, Output_dir)

        #load_images_with_true_label(Input_dir)