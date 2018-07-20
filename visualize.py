# ---------------------------------------------------------------------------------------
#  Visualize optimum stimuli for nodes within the network
#
#  Ref = http://nbviewer.jupyter.org/github/tensorflow/tensorflow/blob/master/tensorflow/
#        examples/tutorials/deepdream/deepdream.ipynb
# ---------------------------------------------------------------------------------------
from __future__ import absolute_import
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


import model.hed as m


def get_layer_cb(layer):
    g = tf.get_default_graph()
    return g.get_tensor_by_name(layer + ':0')


def normalize(img, std=0.1):
    """
    Normalize an image to make it easier to display"
    :param img:
    :param std:
    :return:
    """
    n_img = (img - img.mean()) / max(img.std(), 1e-4) * std + 0.5
    # n_img = (img - img.min()) / (img.max() - img.min())

    return np.uint8(n_img)


def gradient_ascent(tgt_layer_cb, in_cb, start_img, n_iterations=2000, step=0.1):
    t_score = tf.reduce_mean(tgt_layer_cb)
    t_grad = tf.gradients(t_score, in_cb)[0]

    img = start_img.copy()

    for i in range(n_iterations):
        g, score = sess.run([t_grad, t_score], {in_cb: img})
        g /= (g.std() + 1e-8)

        img += g * step
        print("iteration {}, score {}".format(i, score))

    plt.figure()
    img = img.squeeze()
    plt.imshow(normalize(img))


if __name__ == '__main__':
    plt.ion()

    noise_image = np.random.uniform(size=(256, 256, 3)) * 255.0
    #noise_image = np.uint8(noise_image)
    # plt.figure()
    # plt.imshow(noise_image)
    # plt.title("Starting Image")

    model = m.KitModel(weight_file='./model/hed.npy')
    model_input = np.expand_dims(noise_image, 0)

    # # --------------------------------------------------------------------------------
    # # Display Convolution Layers names and number of kernels
    # # --------------------------------------------------------------------------------
    # with tf.Session() as sess:
    #     graph = tf.get_default_graph()
    #
    #     layers = [op.name for op in graph.get_operations() if op.type == 'Conv2D']
    #     feature_nums = [int(graph.get_tensor_by_name(name + ':0').get_shape()[-1]) for name in layers]
    #
    #     print("Total number of Convolutional Layers {}".format(len(layers)))
    #     for l_idx, l in enumerate(layers):
    #         print("Layer {} has {} kernels".format(l, feature_nums[l_idx]))
    #

    # --------------------------------------------------------------------------------
    # Image Space Gradient Ascent
    # --------------------------------------------------------------------------------
    tgt_layer_name = 'convolution_4'
    tgt_neuron_channel = 1

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        input_cb, output_cb = model

        gradient_ascent(
            tgt_layer_cb=get_layer_cb(tgt_layer_name)[:, 20, 20, tgt_neuron_channel],
            in_cb=input_cb,
            start_img=model_input
        )
    raw_input("press any key to exit")
