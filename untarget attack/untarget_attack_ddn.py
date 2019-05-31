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

from typing import Tuple, Optional, Callable
import tensorflow as tf
import numpy as np
import time
begin_time = time.time()

os.environ['CUDA_VISIBLE_DEVICES']='4'

slim = tf.contrib.slim

tf.flags.DEFINE_string(
    'checkpoint_path', '', 'Path to checkpoint for inception network.')
tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')
tf.flags.DEFINE_string(
    'output_dir', '', 'Output directory with images.')

FLAGS = tf.flags.FLAGS

# 声明一些攻击参vi
#CHECKPOINTS_DIR = './model/'
CHECKPOINTS_DIR = FLAGS.checkpoint_path
model_checkpoint_map = {
    'inception_v1': os.path.join(CHECKPOINTS_DIR, 'inception_v1', 'inception_v1.ckpt'),
    'resnet_v1_50': os.path.join(CHECKPOINTS_DIR, 'resnet_v1_50', 'model.ckpt-49800'),
    'vgg_16': os.path.join(CHECKPOINTS_DIR, 'vgg_16', 'vgg_16.ckpt')}

#input_dir = './dev_data/'
#output = './output/'

batch_size = 22



def cosine_distance(x1, x2, eps=1e-8):
    numerator = tf.reduce_sum(x1 * x2, axis=1)
    denominator = tf.norm(x1, axis=1) * tf.norm(x2, axis=1) + eps
    return tf.reduce_mean(numerator / denominator)


def quantization(x, levels):
    return tf.round(x * (levels - 1)) / (levels - 1)


class DDN_tf:
    """
    DDN attack: decoupling the direction and norm of the perturbation to
    achieve a small L2 norm in few steps.

    Parameters
    ----------
    model : Callable
        A function that accepts a tf.placeholder as argument, and returns
        logits (pre-softmax activations)
    batch_shape : tuple (B x H x W x C)
        The input shape
    steps : int
        Number of steps for the optimization.
    targeted : bool
        Whether to perform a targeted attack or not.
    gamma : float, optional
        Factor by which the norm will be modified:
            new_norm = norm * (1 + or - gamma).
    init_norm : float, optional
        Initial value for the norm.
    quantize : bool, optional
        If True, the returned adversarials will have quantized values to the
         specified number of levels.
    levels : int, optional
        Number of levels to use for quantization (e.g. 256 for 8 bit images).
    max_norm : float or None, optional
        If specified, the norms of the perturbations will not be greater than
        this value which might lower success rate.
    callback : object, optional
        Visdom callback to display various metrics.

    """

    def __init__(self, model: Callable, batch_shape: Tuple[int, int, int, int],
                 steps: int, targeted: bool, gamma: float = 0.05,
                 init_norm: float = 1., quantize: bool = True,
                 levels: int = 256, max_norm: float or None = None,
                 callback: Optional = None) -> None:
        self.steps = steps
        self.max_norm = max_norm
        self.quantize = quantize
        self.levels = levels
        self.callback = callback

        multiplier = 1 if targeted else -1

        # We keep the images under attack in memory using tf.Variable
        self.inputs = tf.Variable(np.zeros(batch_shape), dtype=tf.float32, name='inputs')
        self.labels = tf.Variable(np.zeros(batch_shape[0]), dtype=tf.int64, name='labels')
        self.assign_inputs = tf.placeholder(tf.float32, batch_shape)
        self.assign_labels = tf.placeholder(tf.int64, batch_shape[0])
        self.setup = [self.inputs.assign(self.assign_inputs),
                      self.labels.assign(self.assign_labels)]

        # Constraints on delta, such that the image remains in [0, 1]
        boxmin = 0 - self.inputs
        boxmax = 1 - self.inputs
        self.worst_norm = tf.norm(tf.layers.flatten(tf.maximum(self.inputs, 1 - self.inputs)), axis=1)

        # delta: the distortion (adversarial noise)
        delta = tf.Variable(np.zeros(batch_shape, dtype=np.float32), name='delta')

        # norm: the current \epsilon-ball around the inputs, on which the attacks are projected
        norm = tf.Variable(np.full(batch_shape[0], init_norm, dtype=np.float32), name='norm')
        self.mean_norm = tf.reduce_mean(norm)

        self.best_delta = tf.Variable(delta)

        adv_found = tf.Variable(np.full(batch_shape[0], 0, dtype=np.bool))
        self.mean_adv_found = tf.reduce_mean(tf.to_float(adv_found))

        self.best_l2 = tf.Variable(self.worst_norm)
        self.mean_best_l2 = tf.reduce_sum(self.best_l2 * tf.to_float(adv_found)) / tf.reduce_sum(tf.to_float(adv_found))

        self.init = tf.variables_initializer(var_list=[delta, norm, self.best_l2, self.best_delta, adv_found])

        # Forward propagation
        adv = self.inputs + delta
        logits = model(adv)
        pred_labels = tf.argmax(logits, 1)
        self.ce_loss = tf.losses.sparse_softmax_cross_entropy(labels=self.labels, logits=logits,
                                                              reduction=tf.losses.Reduction.SUM)

        self.loss = multiplier * self.ce_loss
        if targeted:
            self.is_adv = tf.equal(pred_labels, self.labels)
        else:
            self.is_adv = tf.not_equal(pred_labels, self.labels)

        delta_flat = tf.layers.flatten(delta)
        l2 = tf.norm(delta_flat, axis=1)
        self.mean_l2 = tf.reduce_mean(l2)

        new_adv_found = tf.logical_or(self.is_adv, adv_found)
        self.update_adv_found = tf.assign(adv_found, new_adv_found)
        is_smaller = tf.less(l2, self.best_l2)
        is_both = tf.logical_and(self.is_adv, is_smaller)
        new_best_l2 = tf.where(is_both, l2, self.best_l2)
        self.update_best_l2 = tf.assign(self.best_l2, new_best_l2)
        new_best_delta = tf.where(is_both, delta, self.best_delta)
        self.update_best_delta = tf.assign(self.best_delta, new_best_delta)

        self.update_saved = tf.group(self.update_adv_found, self.update_best_l2, self.update_best_delta)

        # Expand or contract the norm depending on whether the current examples are adversarial
        new_norm = norm * (1 - (2 * tf.to_float(self.is_adv) - 1) * gamma)
        new_norm = tf.minimum(new_norm, self.worst_norm)

        self.step = tf.placeholder(tf.int32, name='step')

        lr = tf.train.cosine_decay(learning_rate=1., global_step=self.step, decay_steps=steps, alpha=0.01)
        self.lr = tf.reshape(lr, ())  # Tensorflow doesnt know its shape.

        # Compute the gradient and renorm it
        grad = tf.gradients(self.loss, delta)[0]
        grad_flat = tf.layers.flatten(grad)

        grad_norm_flat = tf.norm(grad_flat, axis=1)
        grad_norms = tf.reshape(grad_norm_flat, (-1, 1, 1, 1))
        new_grad = grad / grad_norms

        # Corner case: if gradient is zero, take a random direction
        is_grad_zero = tf.equal(grad_norm_flat, 0)
        random_values = tf.random_normal(batch_shape)

        grad_without_zeros = tf.where(is_grad_zero, random_values, new_grad)
        grad_without_zeros_flat = tf.layers.flatten(grad_without_zeros)

        # Take a step in the gradient direction
        new_delta = delta - self.lr * grad_without_zeros

        new_l2 = tf.norm(tf.layers.flatten(new_delta), axis=1)
        normer = tf.reshape(new_norm / new_l2, (-1, 1, 1, 1))
        new_delta = new_delta * normer

        if quantize:
            # Quantize delta (e.g. such that the resulting image has 256 values)
            new_delta = quantization(new_delta, levels)

        # Ensure delta is on the valid range
        new_delta = tf.clip_by_value(new_delta, boxmin, boxmax)
        self.update_delta = tf.assign(delta, new_delta)
        self.update_norm = tf.assign(norm, new_norm)

        # Update operation (updates both delta and the norm)
        self.update_op = tf.group(self.update_delta, self.update_norm)

        # Cosine between self.delta and new grad
        self.cosine = cosine_distance(-delta_flat, grad_without_zeros_flat)

        # Renorm if max-norm is provided
        if max_norm:
            best_delta_flat = tf.layers.flatten(self.best_delta)
            best_delta_renormed = tf.clip_by_norm(best_delta_flat, max_norm, axes=1)
            if quantize:
                best_delta_renormed = quantization(best_delta_renormed, levels)
            self.best_delta_renormed = tf.reshape(best_delta_renormed, batch_shape)

    def attack(self, sess: tf.Session, inputs: np.ndarray,
               labels: np.ndarray) -> np.ndarray:
        """
        Performs the attack of the model for the inputs and labels.

        Parameters
        ----------
        sess : tf session
            Tensorflow session
        inputs : np.ndarray
            Batch of samples to attack. Values should be in the [0, 1] range.
        labels : np.ndarray
            Labels of the samples to attack if untargeted,
            else labels of targets.

        Returns
        -------
        np.ndarray
            Batch of samples modified to be adversarial to the model.

        """
        if inputs.min() < 0 or inputs.max() > 1:
            raise ValueError('Input values should be in the [0, 1] range.')

        sess.run(self.setup, feed_dict={self.assign_inputs: inputs, self.assign_labels: labels})
        sess.run(self.init)
        for i in range(self.steps):
            # Runs one step and collects statistics
            if self.callback:
                results = sess.run([self.ce_loss, self.mean_l2, self.mean_norm, self.cosine, self.update_saved])
                loss, l2, norm, cosine, _, = results
                best_l2, adv_found = sess.run([self.mean_best_l2, self.mean_adv_found])
            else:
                sess.run(self.update_saved)

            lr, _ = sess.run([self.lr, self.update_op], feed_dict={self.step: i})

            if self.callback:
                self.callback.scalar('ce', i, loss / len(inputs))
                self.callback.scalars(['max_norm', 'l2', 'best_l2'], i,
                                      [norm, l2, best_l2 if adv_found else norm])
                self.callback.scalars(['cosine', 'lr', 'success'], i, [cosine, lr, adv_found])

        if self.max_norm:
            best_delta = sess.run(self.best_delta_renormed)
        else:
            best_delta = sess.run(self.best_delta)

        return inputs + best_delta


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

    #input remains in[0,1]
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




def attack_ddn(input_dir, output_dir):

    batch_shape = [batch_size, 224, 224, 3]
    check_or_create_dir(output_dir)

    with tf.Graph().as_default():
        # Prepare graph
        attacker = DDN_tf(model=ens_model, batch_shape=batch_shape, steps=100, targeted=True)
        # preprocessing for model input,
        # note that images for all classifier will be normalized to be in [0, 1]

        #     y = tf.constant(np.zeros([batch_size]), tf.int64)

        # Run computation
        s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV1'))
        s2 = tf.train.Saver(slim.get_model_variables(scope='resnet_v1_50'))
        s3 = tf.train.Saver(slim.get_model_variables(scope='vgg_16'))

        with tf.Session() as sess:
            s1.restore(sess, model_checkpoint_map['inception_v1'])
            s2.restore(sess, model_checkpoint_map['resnet_v1_50'])
            s3.restore(sess, model_checkpoint_map['vgg_16'])

            for filenames, raw_images, target_labels in load_images_with_target_label(input_dir):
                #processed_imgs_ = sess.run(processed_imgs, feed_dict={raw_inputs: raw_images})
                adv_images = attacker.attack(sess, raw_images, target_labels)
                save_images(adv_images, filenames, output_dir)




if __name__ == '__main__':
    #input_dir = 'dev_data'
    #output_dir = 'output_ddn'
    input_dir = FLAGS.input_dir
    output_dir = FLAGS.output_dir
    attack_ddn(input_dir, output_dir)
    print('Time cost is %f s' % (time.time() - begin_time))
    pass