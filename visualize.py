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


def get_number_of_channels(tgt_l, model_graph):
    """
    Get the number of channels in a specified layer
    :param tgt_l:
    :param model_graph:
    :return:
    """
    return int(model_graph.get_tensor_by_name(tgt_l + ':0').get_shape()[-1])


def visualize_all_channels(img, model_graph, in_cb, tgt_l, margin=3, n_iter=50):
    """

    :param img:
    :param model_graph:
    :param in_cb:
    :param tgt_l:
    :param margin:
    :param n_iter:
    :return:
    """

    n_channels = get_number_of_channels(tgt_l, model_graph)
    print("{} has {} channels".format(tgt_l, n_channels))

    tile_d = np.int(np.ceil(np.sqrt(n_channels)))  # Single dimension of tiled image

    r = img.shape[1]
    c = img.shape[2]

    width = (r + margin) * tile_d
    height = (c + margin) * tile_d

    tiled_img = np.zeros((width, height, img.shape[3]))

    for tgt_chan in range(n_channels):

        r_idx = tgt_chan / tile_d
        c_idx = tgt_chan - r_idx * tile_d
        print("processing channel {}, (r,c)= {},{}".format(tgt_chan, r_idx, c_idx))

        processed_img = simple_gradient_ascent(
            tgt_layer_cb[:, :, :, tgt_chan],
            in_cb,
            img,
            n_iter=n_iter,
        )

        tiled_img[
            r_idx * (r + margin): r_idx * (r + margin) + r,
            c_idx * (c + margin): c_idx * (c + margin) + c,
            :
        ] = processed_img

    display_image(tiled_img)


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
    tgt_layer = 'block_1/convolution_1'
    # tgt_layer = 'block_7/conv2d_transpose'

    # Start with a gray image with noise
    start_image = np.random.uniform(size=(256, 256, 3)) + 100.
    start_image = np.expand_dims(start_image, 0)

    tgt_layer_cb = get_layer_callback(tgt_layer, graph)

    # -----------------------------------------------------------------------------------
    # Single Channel of specified layer
    # -----------------------------------------------------------------------------------
    tgt_channel = 3

    processed_image = simple_gradient_ascent(
        tgt_layer_cb[:, 100, 100, tgt_channel],
        input_cb,
        start_image,
        n_iter=100,
    )

    display_image(vis_normalize(processed_image))
    plt.title("Simple Gradient Ascent")

    # # -----------------------------------------------------------------------------------
    # # All Channels of specified Layer
    # # -----------------------------------------------------------------------------------
    # visualize_all_channels(
    #     start_image,
    #     graph,
    #     input_cb,
    #     tgt_layer
    # )
    #
    # plt.title("All {} channels of layer {}".format(
    #     get_number_of_channels(tgt_layer, graph),
    #     tgt_layer)
    # )

    # -----------------------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------------------
    raw_input("Press any Key to Exit")
