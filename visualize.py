# ---------------------------------------------------------------------------------------
#  Find image that maximizes activations of a specified neuron/layer
#
#  Ref = http://nbviewer.jupyter.org/github/tensorflow/tensorflow/blob/master/tensorflow/
#        examples/tutorials/deepdream/deepdream.ipynb
# ---------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

import model.hed as m


def get_layer_callback(l_name, model_graph):
    """
    Helper function for getting layer output tensor
    :param l_name:
    :param model_graph:
    :return:
    """
    return model_graph.get_tensor_by_name("{}:0".format(l_name))


def vis_normalize(a, s=0.1):
    """
    Normalize a  image for display
    :param s:
    :param a:
    :return:
    """
    return s * (a - a.mean()) / (max(a.std(), 1e-4)) + 0.5


def display_image(a, axis=None):
    if axis is None:
        _, axis = plt.subplots()

    a = np.uint8(np.clip(a, 0, 1) * 255.0)
    axis.imshow(a)


def simple_gradient_ascent(tgt_cb, in_cb, in_img, n_iter=20, step=1.0):

    t_score = tf.reduce_mean(tgt_cb)  # Reduce the mean activation of the output
    t_grad = tf.gradients(t_score, in_cb)[0]  # wrt to the input variable

    img = in_img.copy()
    with tf.Session() as s:

        s.run(tf.global_variables_initializer())

        for i in range(n_iter):
            g, score = s.run([t_grad, t_score], {in_cb: img})

            # normalizing the gradient, so the same step size should work for different layers and networks
            # for different layers and networks
            g /= g.std() + 1e-8
            img += g * step
            print("{}:Score {}".format(i, score))

    # Display the image
    img = np.squeeze(img, axis=0)
    return vis_normalize(img)


if __name__ == '__main__':
    plt.ion()
    np.random.seed(7)

    # -----------------------------------------------------------------------------------
    # Load the Model
    # -----------------------------------------------------------------------------------
    input_cb, output_cb = m.KitModel(weight_file='./model/hed.npy')
    graph = tf.get_default_graph()

    # -----------------------------------------------------------------------------------
    # Display Model Architecture
    # -----------------------------------------------------------------------------------
    with tf.Session() as sess:
        LOGDIR = './logs'
        train_writer = tf.summary.FileWriter(LOGDIR)
        train_writer.add_graph(sess.graph)

    # To view:
    # [1] tensorboard --logdir=./logs
    # [2] In a browser window open the printed address

    # # ---------------------------------------------------------------------------
    # # Print all the Convolutional/Deconvolution Layers in the model and their dimensions
    # # ---------------------------------------------------------------------------
    # with tf.Session() as sess:
    #
    #     layers = [op.name for op in graph.get_operations() if
    #               (op.type == 'Conv2D' or op.type == 'Conv2DBackpropInput')]
    #     feature_nums = [int(graph.get_tensor_by_name(name + ':0').get_shape()[-1]) for name in layers]
    #
    #     print("Total Number of layers {}".format(len(layers)))
    #     for l_idx, layer in enumerate(layers):
    #         print("{} has {} kernels".format(layer, feature_nums[l_idx]))

    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    tgt_layer = 'block_4/convolution'
    # tgt_layer = 'block_7/conv2d_transpose'

    # Start with a gray image with noise
    start_image = np.random.uniform(size=(256, 256, 3)) + 100.
    start_image = np.expand_dims(start_image, 0)

    tgt_layer_cb = get_layer_callback(tgt_layer, graph)

    # -----------------------------------------------------------------------------------
    # Single Channel of specified layer
    # -----------------------------------------------------------------------------------
    tgt_channel = 0

    processed_image = simple_gradient_ascent(
        tgt_layer_cb[:, :, :, tgt_channel],
        input_cb,
        start_image,
        n_iter=100,
    )

    display_image(vis_normalize(processed_image))
    plt.title("Simple Gradient Ascent")

    # # -----------------------------------------------------------------------------------
    # # All Channels of specified Layer
    # # -----------------------------------------------------------------------------------
    # n_channels = int(graph.get_tensor_by_name(tgt_layer + ':0').get_shape()[-1])
    # print("{} has {} channels".format(tgt_layer, n_channels))
    #
    # tile_d = np.int(np.ceil(np.sqrt(n_channels)))  # Single dimension of tiled image
    # tile_margin = 3
    #
    # r = start_image.shape[1]
    # c = start_image.shape[2]
    # #
    # # width = (tile_d * r) + ((tile_d - 1) * tile_margin)
    # # height = (tile_d * c) + ((tile_d - 1) * tile_margin)
    #
    # width = (r + tile_margin) * tile_d
    # height = (c+tile_margin) * tile_d
    #
    # tiled_image = np.zeros((width, height, start_image.shape[3]))
    #
    # for tgt_channel in range(n_channels):
    #
    #     r_idx = tgt_channel / tile_d
    #     c_idx = tgt_channel - r_idx * tile_d
    #     print("processing channel {}, (r,c)= {},{}".format(tgt_channel, r_idx, c_idx))
    #
    #     processed_img = simple_gradient_ascent(
    #         tgt_layer_cb[:, :, :, tgt_channel],
    #         input_cb,
    #         start_image,
    #         n_iter=50,
    #     )
    #
    #     # print ("(r {},{})".format(r_idx * (r + tile_margin), (r_idx + 1) * (r + tile_margin)))
    #     # print ("(c {},{})".format(c_idx * (c + tile_margin), (c_idx + 1) * (c + tile_margin)))
    #
    #     tiled_image[
    #         r_idx * (r + tile_margin): r_idx * (r + tile_margin) + r,
    #         c_idx * (c + tile_margin): c_idx * (c + tile_margin) + c,
    #         :
    #     ] = processed_img
    #
    # display_image(tiled_image)

    # -----------------------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------------------
    raw_input("Press any Key to Exit")
