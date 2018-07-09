import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from model.hed import KitModel
from border.stimuli import Colours, get_image, add_rectangle


def find_optimal_bars(input, layers):
    """
    Finds bar stimuli that optimally activate each of the feature maps in a single layer of a
    convolutional network, approximating the procedure in:

    H. Zhou, H. S. Friedman, and R. von der Heydt, “Coding of border ownership in monkey visual
    cortex.,” J. Neurosci., vol. 20, no. 17, pp. 6594–6611, 2000.

    Their description of the procedure is, ""After isolation of a cell, the receptive field was
    examined with rectangular bars, and the optimal stimulus parameters were
    determined by varying the length, width, color, orientation ..."

    We approximate this by applying a variety of bar stimuli, and finding which one most strongly
    activates the centre unit in each feature map. Testing the whole layer at once is more
    efficient than testing each feature map individually, since the whole network up to that point
    must be run whether we record a single unit or all of them.

    :param input: Input to TensorFlow model (Placeholder node)
    :param layers: Layers of convolutional network to record from
    :return: parameters, responses, preferred_stimuli
    """

    colours = Colours()

    bg_colour_name = 'Light gray (background)'
    bg_colour = colours.get_RGB(bg_colour_name, 0)

    fg_colour_names = [key for key in colours.colours.keys()
                       if key != bg_colour_name]

    # TODO: probably need more sizes and angles, also shift bar laterally
    lengths = [40, 80]
    widths = [4, 8]
    angles = np.pi * np.array([0, .25, .5, .75])

    parameters = []
    responses = {}
    preferred_stimuli = {}

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        model_tf = ()
        for layer in layers:
            model_tf += (sess.graph.get_tensor_by_name(layer),)
            responses[layer] = []

        for fg_colour_name in fg_colour_names:
            input_data = None
            n_luminances = colours.get_num_luminances(fg_colour_name)
            n_stimuli = len(lengths) * len(widths) * len(angles) * n_luminances
            print('Testing {} {} stimuli'.format(n_stimuli, fg_colour_name))
            for i in range(n_luminances):
                RGB = colours.get_RGB(fg_colour_name, i)
                for length in lengths:
                    for width in widths:
                        for angle in angles:
                            parameters.append({
                                'colour': RGB,
                                'length': length,
                                'width': width,
                                'angle': angle})

                            stimulus = get_image((400, 400, 3), bg_colour)
                            add_rectangle(stimulus, (200, 200),
                                          (width, length), angle, RGB)

                            # plt.imshow(stimulus)
                            # plt.show()

                            if input_data is None:
                                input_data = np.expand_dims(stimulus, 0)
                            else:
                                input_data = np.concatenate(
                                    (input_data, np.expand_dims(stimulus, 0)), 0)

            activities = sess.run(
                model_tf, feed_dict={input: input_data})
            # activities is a tuple with shape stim x h x w x feats
            for i, activity in enumerate(activities):
                centre = (
                    int(activity.shape[1] / 2), int(activity.shape[2] / 2))

                responses[layers[i]].append(
                    activity[:, centre[0], centre[1], :])

    for layer in layers:
        # reshape to layers x stim x feats
        responses[layer] = np.concatenate(responses[layer])
        preferred_stimuli[layer] = np.argmax(responses[layer], axis=0)

    return parameters, responses, preferred_stimuli


if __name__ == '__main__':
    model_converted = KitModel('model/hed.npy')
    input_tf, _ = model_converted
    # Define layers to perform experiment on
    layers = ['relu1_1', 'relu1_2', 'relu2_1', 'relu2_2', 'relu3_1', 'relu3_2',
              'relu3_3', 'relu4_1', 'relu4_2', 'relu4_3', 'relu5_1', 'relu5_2', 'relu5_3']
    device_append = ':0'
    layers = [layer + device_append for layer in layers]

    parameters, responses, preferred_stimuli = find_optimal_bars(
        input_tf, layers)

    # print(parameters)
    # print(responses)
    # print(preferred_stimuli)
