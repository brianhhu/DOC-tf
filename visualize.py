# ---------------------------------------------------------------------------------------
#  Find image that maximizes activations of a specified neuron/layer
#
#  Ref = http://nbviewer.jupyter.org/github/tensorflow/tensorflow/blob/master/tensorflow/
#        examples/tutorials/deepdream/deepdream.ipynb
# ---------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

import model.hed as model_hed
import model.doc as model_doc


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
            print("{}: Score {}".format(i, score))

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


def visualize_all_channels(img, model_graph, in_cb, tgt_l,  n_iter=50, chans_per_image=32, margin=1):
    """

    :param img:
    :param model_graph:
    :param in_cb:
    :param tgt_l:
    :param n_iter:
    :param chans_per_image:
    :param margin:
    :return:
    """

    n_channels = get_number_of_channels(tgt_l, model_graph)
    print("{} has {} channels".format(tgt_l, n_channels))

    tile_d = np.int(np.ceil(np.sqrt(chans_per_image)))  # Single dimension of tiled image

    r = img.shape[1]
    c = img.shape[2]

    width = (r + margin) * tile_d
    height = (c + margin) * tile_d

    img_idx = 0
    tiled_img = np.zeros((width, height, img.shape[3]))

    for tgt_chan in range(n_channels):

        temp = np.mod(tgt_chan, chans_per_image)

        if temp == 0:
            tiled_img = np.zeros((width, height, img.shape[3]))
            img_idx += 1

        r_idx = temp / tile_d
        c_idx = temp - r_idx * tile_d
        print("processing channel {}, image {}, (r,c)= {},{}".format(tgt_chan, img_idx, r_idx, c_idx))

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

        if temp == (chans_per_image - 1):
            display_image(tiled_img)
            plt.title("Layer {}. Channels {} - {}".format(
                tgt_l,
                (img_idx - 1) * chans_per_image,
                (img_idx * chans_per_image) - 1,
            ))


def visualize_max_neuron_activation(t_layer_name, t_chan, t_loc, g, n_iter=100):
    """

    :param t_layer_name:
    :param t_chan:
    :param t_loc:
    :param g: graph
    :param n_iter:

    :return:
    """
    t_layer_cb = get_layer_callback(t_layer_name, g)
    print("Target Layer {}. shape {}".format(t_layer_cb.name, t_layer_cb.get_shape()))

    # Start with a gray image with noise
    start_img = np.random.uniform(size=(256, 256, 3)) + 100.
    start_img = np.expand_dims(start_img, 0)

    final_img = simple_gradient_ascent(
        t_layer_cb[:, t_loc, t_loc, t_chan],
        input_cb,
        start_img,
        n_iter=n_iter,
    )

    display_image(vis_normalize(final_img))
    plt.title("Layer name {}. Channel Index {}. Neuron at index ({},{})".format(
        t_layer_name, t_chan, t_loc, t_loc))


if __name__ == '__main__':
    plt.ion()
    np.random.seed(7)

    # # ***********************************************************************************
    # #  HED Model
    # # ***********************************************************************************
    # print("Analyzing HED Network")
    #
    # # Load the Model
    # input_cb, output_cb = model_hed.KitModel(weight_file='./model/hed.npy')
    # graph = tf.get_default_graph()
    #
    # with tf.Session() as sess:
    #     tensorboard_logs_dir = './logs'
    #     train_writer = tf.summary.FileWriter(tensorboard_logs_dir)
    #     train_writer.add_graph(sess.graph)
    #
    #     # To view:
    #     # [1] tensorboard --logdir=./logs
    #     # [2] In a browser window, open http://localhost:6006/
    #
    # # List Layer names of interest
    # with tf.Session() as sess:
    #
    #     layers = [op.name for op in graph.get_operations() if ((op.type == 'Conv2D') or (op.type == 'Relu'))]
    #
    #     feature_nums = [int(graph.get_tensor_by_name(name + ':0').get_shape()[-1]) for name in layers]
    #
    #     print("Total Number of layers {}".format(len(layers)))
    #     for l_idx, layer in enumerate(layers):
    #         print("{} has {} kernels".format(layer, feature_nums[l_idx]))
    #
    # # Visualize max activations of target cells
    # # Good Border cells
    # tgt_layer = 'convolution_12'  # the convolution operation right before relu5_3
    # tgt_channel = 441
    # center_neuron_idx = 8  # for size 256, 256
    #
    # visualize_max_neuron_activation(tgt_layer, tgt_channel, center_neuron_idx, graph)
    # plt.suptitle("Good Border Cell")
    #
    # # Good Contrast Cells
    # tgt_layer = 'convolution_12'  # the convolution operation right before relu5_3
    # tgt_channel = 88
    # center_neuron_idx = 8  # for size 256, 256
    #
    # visualize_max_neuron_activation(tgt_layer, tgt_channel, center_neuron_idx, graph)
    # plt.suptitle("Good Contrast Cell")
    #
    # tgt_layer = 'convolution_12'  # the convolution operation right before relu5_3
    # tgt_channel = 130
    # center_neuron_idx = 8  # for size 256, 256
    #
    # visualize_max_neuron_activation(tgt_layer, tgt_channel, center_neuron_idx, graph)
    # plt.suptitle("Good Contrast Cell")
    #
    # tgt_layer = 'convolution_12'  # the convolution operation right before relu5_3
    # tgt_channel = 154
    # center_neuron_idx = 8  # for size 256, 256
    #
    # visualize_max_neuron_activation(tgt_layer, tgt_channel, center_neuron_idx, graph)
    # plt.suptitle("Good Contrast Cell")
    #
    # tgt_layer = 'convolution_12'  # the convolution operation right before relu5_3
    # tgt_channel = 170
    # center_neuron_idx = 8  # for size 256, 256
    #
    # visualize_max_neuron_activation(tgt_layer, tgt_channel, center_neuron_idx, graph)
    # plt.suptitle("Good Contrast Cell")
    #
    # tgt_layer = 'convolution_12'  # the convolution operation right before relu5_3
    # tgt_channel = 314
    # center_neuron_idx = 8  # for size 256, 256
    #
    # visualize_max_neuron_activation(tgt_layer, tgt_channel, center_neuron_idx, graph)
    # plt.suptitle("Good Contrast Cell")
    #
    # tgt_layer = 'convolution_12'  # the convolution operation right before relu5_3
    # tgt_channel = 464
    # center_neuron_idx = 8  # for size 256, 256
    #
    # visualize_max_neuron_activation(tgt_layer, tgt_channel, center_neuron_idx, graph)
    # plt.suptitle("Good Contrast Cell")
    #
    # tgt_layer = 'convolution_12'  # the convolution operation right before relu5_3
    # tgt_channel = 488
    # center_neuron_idx = 8  # for size 256, 256
    #
    # visualize_max_neuron_activation(tgt_layer, tgt_channel, center_neuron_idx, graph)
    # plt.suptitle("Good Contrast Cell")

    # ***********************************************************************************
    #  DOC Model
    # ***********************************************************************************
    print("Analyzing DOC Network")

    # Load the Model
    input_cb, output_cb = model_doc.KitModel(weight_file='./model/doc.npy')
    graph = tf.get_default_graph()

    with tf.Session() as sess:
        tensorboard_logs_dir = './logs'
        train_writer = tf.summary.FileWriter(tensorboard_logs_dir)
        train_writer.add_graph(sess.graph)

        # To view:
        # [1] tensorboard --logdir=./logs
        # [2] In a browser window, open http://localhost:6006/

    # # List Layer names of interest
    # with tf.Session() as sess:
    #
    #     layers = [op.name for op in graph.get_operations() if ((op.type == 'Conv2D') or (op.type == 'Relu'))]
    #
    #     feature_nums = [int(graph.get_tensor_by_name(name + ':0').get_shape()[-1]) for name in layers]
    #
    #     print("Total Number of layers {}".format(len(layers)))
    #     for l_idx, layer in enumerate(layers):
    #         print("{} has {} kernels".format(layer, feature_nums[l_idx]))

    # Visualize Target Cells
    # Good Border cells
    tgt_layer = 'convolution_12'  # the convolution operation right before relu5_3
    tgt_channel = 121
    center_neuron_idx = 8  # for size 256, 256

    visualize_max_neuron_activation(tgt_layer, tgt_channel, center_neuron_idx, graph)
    plt.suptitle("Good Border Cell")

    tgt_layer = 'convolution_12'  # the convolution operation right before relu5_3
    tgt_channel = 204
    center_neuron_idx = 8  # for size 256, 256

    visualize_max_neuron_activation(tgt_layer, tgt_channel, center_neuron_idx, graph)
    plt.suptitle("Good Border Cell")

    tgt_layer = 'convolution_12'  # the convolution operation right before relu5_3
    tgt_channel = 254
    center_neuron_idx = 8  # for size 256, 256

    visualize_max_neuron_activation(tgt_layer, tgt_channel, center_neuron_idx, graph)
    plt.suptitle("Good Border Cell")

    tgt_layer = 'convolution_12'  # the convolution operation right before relu5_3
    tgt_channel = 326
    center_neuron_idx = 8  # for size 256, 256

    visualize_max_neuron_activation(tgt_layer, tgt_channel, center_neuron_idx, graph)
    plt.suptitle("Good Border Cell")

    tgt_layer = 'convolution_12'  # the convolution operation right before relu5_3
    tgt_channel = 476
    center_neuron_idx = 8  # for size 256, 256

    visualize_max_neuron_activation(tgt_layer, tgt_channel, center_neuron_idx, graph)
    plt.suptitle("Good Border Cell")

    # Good Contrast Cells
    tgt_layer = 'convolution_12'  # the convolution operation right before relu5_3
    tgt_channel = 81
    center_neuron_idx = 8  # for size 256, 256

    visualize_max_neuron_activation(tgt_layer, tgt_channel, center_neuron_idx, graph)
    plt.suptitle("Good Contrast Cell")

    tgt_layer = 'convolution_12'  # the convolution operation right before relu5_3
    tgt_channel = 94
    center_neuron_idx = 8  # for size 256, 256

    visualize_max_neuron_activation(tgt_layer, tgt_channel, center_neuron_idx, graph)
    plt.suptitle("Good Contrast Cell")

    tgt_layer = 'convolution_12'  # the convolution operation right before relu5_3
    tgt_channel = 199
    center_neuron_idx = 8  # for size 256, 256

    visualize_max_neuron_activation(tgt_layer, tgt_channel, center_neuron_idx, graph)
    plt.suptitle("Good Contrast Cell")

    tgt_layer = 'convolution_12'  # the convolution operation right before relu5_3
    tgt_channel = 205
    center_neuron_idx = 8  # for size 256, 256

    visualize_max_neuron_activation(tgt_layer, tgt_channel, center_neuron_idx, graph)
    plt.suptitle("Good Contrast Cell")

    tgt_layer = 'convolution_12'  # the convolution operation right before relu5_3
    tgt_channel = 226
    center_neuron_idx = 8  # for size 256, 256

    visualize_max_neuron_activation(tgt_layer, tgt_channel, center_neuron_idx, graph)
    plt.suptitle("Good Contrast Cell")

    tgt_layer = 'convolution_12'  # the convolution operation right before relu5_3
    tgt_channel = 328
    center_neuron_idx = 8  # for size 256, 256

    visualize_max_neuron_activation(tgt_layer, tgt_channel, center_neuron_idx, graph)
    plt.suptitle("Good Contrast Cell")

    tgt_layer = 'convolution_12'  # the convolution operation right before relu5_3
    tgt_channel = 491
    center_neuron_idx = 8  # for size 256, 256

    visualize_max_neuron_activation(tgt_layer, tgt_channel, center_neuron_idx, graph)
    plt.suptitle("Good Contrast Cell")

    # -----------------------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------------------
    input("Press any Key to Exit")
