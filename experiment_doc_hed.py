import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import model.doc as doc
import model.hed as hed
from border.stimuli import Colours, get_image, add_rectangle


def find_optimal_bars(doc_converted, hed_converted):
    """
    Finds bar stimuli that optimally activate the DOC and HED networks, approximating the procedure in:

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
    layers = ['doc', 'hed']
    responses = {'doc': [], 'hed': []}
    preferred_stimuli = {}

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        # Get input and output tensors
        doc_input, doc_output = doc_converted
        # only find optimal stimulus on hed network?
        hed_input, hed_output = hed_converted

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

            activities = sess.run((doc_output, hed_output), feed_dict={
                                  doc_input: input_data, hed_input: input_data})

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


def standard_test(doc_converted, hed_converted, preferred_stimulus):
    # Note: we put edge of square on centre of preferred-stimulus bar
    # Zhou et al. determined significance of the effects of contrast and border ownership with
    # a 3-factor ANOVA, significance .01. The factors were side-of-ownership, contrast polarity,
    # and time. Having no time component we use a two-factor ANOVA.
    # "In the standard test, sizes
    # of 4 or 6° were used for cells of V1 and V2, and sizes between 4 and 17°
    # were used for cells of V4, depending on response field size."
    # I don't see where they mention the number of reps per condition, but there are 10 reps
    # in Figure 4.

    colours = Colours()
    bg_colour_name = 'Light gray (background)'
    bg_colour = colours.get_RGB(bg_colour_name, 0)

    # Use contour response
    preferred_stimulus = parameters[preferred_stimulus['hed'][0]]
    preferred_colour = preferred_stimulus['colour']

    square_shape = (100, 100)

    angle = preferred_stimulus['angle']
    rotation = [[np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]]
    position_1 = np.add(np.dot(rotation, np.array(
        [-50, 0]).transpose()), [200, 200]).astype(np.int)
    position_2 = np.add(np.dot(rotation, np.array([50, 0]).transpose()), [
                        200, 200]).astype(np.int)

    # preferred_shape = (preferred_stimulus['width'], preferred_stimulus['length'])
    # add_rectangle(stimulus, (200,200), preferred_shape, angle, preferred_colour)

    # Stimuli as in panels A-D of Zhou et al. Figure 2
    stimulus_A = get_image((400, 400, 3), preferred_colour)
    add_rectangle(stimulus_A, position_1, square_shape, angle, bg_colour)

    stimulus_B = get_image((400, 400, 3), bg_colour)
    add_rectangle(stimulus_B, position_2, square_shape,
                  angle, preferred_colour)

    stimulus_C = get_image((400, 400, 3), bg_colour)
    add_rectangle(stimulus_C, position_1, square_shape,
                  angle, preferred_colour)

    stimulus_D = get_image((400, 400, 3), preferred_colour)
    add_rectangle(stimulus_D, position_2, square_shape, angle, bg_colour)

    input_data = np.stack((stimulus_A, stimulus_B, stimulus_C, stimulus_D))

#     print(input_data.shape)
    fig = plt.figure(figsize=(12, 8))
    plt.subplot(2, 6, 1)
    plt.imshow(stimulus_A)
    plt.title('A')
    plt.subplot(2, 6, 2)
    plt.imshow(stimulus_C)
    plt.title('C')
    plt.subplot(2, 6, 7)
    plt.imshow(stimulus_B)
    plt.title('B')
    plt.subplot(2, 6, 8)
    plt.imshow(stimulus_D)
    plt.title('D')
#     plt.show()

    layers = ['doc', 'hed']
    responses = {'doc': [], 'hed': []}
    side = {'doc': [], 'hed': []}
    contrast = {'doc': [], 'hed': []}
    m = {'doc': [], 'hed': []}

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        # Get input and output tensors
        doc_input, doc_output = doc_converted
        hed_input, hed_output = hed_converted

        activities = sess.run((doc_output, hed_output), feed_dict={
                              doc_input: input_data, hed_input: input_data})

        # Do this for each map separately
        for i, activity in enumerate(activities):
            centre = (int(activity.shape[1] / 2), int(activity.shape[2] / 2))
            responses[layers[i]] = activity[:, centre[0], centre[1]]

            m[layers[i]] = np.mean(responses[layers[i]])
            A, B, C, D = responses[layers[i]]
            side[layers[i]] = np.abs((A+C)/2 - (B+D)/2) / m[layers[i]] * 100
            contrast[layers[i]] = np.abs(
                (A+B)/2 - (C+D)/2) / m[layers[i]] * 100
            # print('side: {} contrast: {}'.format(side, contrast))

    # Orientation
#     plt.figure()
    plt.subplot(2, 6, 3)
    plt.imshow(activities[0][0].squeeze())
    plt.title('A')
    plt.subplot(2, 6, 4)
    plt.imshow(activities[0][2].squeeze())
    plt.title('C')
    plt.subplot(2, 6, 9)
    plt.imshow(activities[0][1].squeeze())
    plt.title('B')
    plt.subplot(2, 6, 10)
    plt.imshow(activities[0][3].squeeze())
    plt.title('D')

    # Contour
#     plt.figure()
    plt.subplot(2, 6, 5)
    plt.imshow(activities[1][0].squeeze())
    plt.title('A')
    plt.subplot(2, 6, 6)
    plt.imshow(activities[1][2].squeeze())
    plt.title('C')
    plt.subplot(2, 6, 11)
    plt.imshow(activities[1][1].squeeze())
    plt.title('B')
    plt.subplot(2, 6, 12)
    plt.imshow(activities[1][3].squeeze())
    plt.title('D')
    plt.tight_layout()
#     fig.savefig('doc_preferred.png', dpi=200)

#     # Show the different conditions
#     plt.figure(figsize=(7,3))
#     plt.subplot(1, 2, 1)
#     plt.bar([0, 1, 2, 3], responses['hed'].squeeze())
#     plt.xticks([0, 1, 2, 3], ('A', 'B', 'C', 'D'))
#     plt.ylabel('BOS')
#     plt.title('HED')

#     plt.subplot(1, 2, 2)
#     plt.bar([0, 1, 2, 3], responses['doc'].squeeze())
#     plt.xticks([0, 1, 2, 3], ('A', 'B', 'C', 'D'))
#     plt.ylabel('BOS')
#     plt.title('DOC')

    return {'responses': responses, 'side': side, 'contrast': contrast, 'mean': m}

    # TODO: the side does involve a contrast difference if the square doesn't fully cover the classical RF,
    #   but as this net is feedforward, there can't be a border effect if it does
    # TODO: generate 10 Poisson random samples from each condition mean for ANOVA
    # TODO: how to set scale for Poisson samples? maybe assume preferred stim evokes a certain rate?


def count_feature_maps(layer):
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        model_tf = sess.graph.get_tensor_by_name(layer)
        result = model_tf.shape[3]

    return result


def standard_test_full_layer(layer, preferred_stimuli):
    # layer = 'relu1_1:0'
    m = count_feature_maps(layer)

    border_responses = []
    contrast_responses = []
    means = []
    for unit_index in range(m):
        print('{} of {} for {}'.format(unit_index, m, layer))
        result = standard_test(input_tf, layer, unit_index,
                               parameters[preferred_stimuli[layer][unit_index]])
        border_responses.append(result['side'])
        contrast_responses.append(result['contrast'])
        means.append(result['mean'])

    with open('border-{}.pkl'.format(layer), 'wb') as file:
        pickle.dump({'border_responses': border_responses,
                     'contrast_responses': contrast_responses,
                     'means': means}, file)

    border_responses = [br for br in border_responses if not np.isnan(br)]
    contrast_responses = [cr for cr in contrast_responses if not np.isnan(cr)]
    print(border_responses)
    print(contrast_responses)

    bins = np.linspace(0, 200, 21)
    plt.figure(figsize=(7,3))
    plt.subplot(1, 3, 1)
    plt.hist(border_responses, bins=bins)
    plt.title('Border Responses')
    plt.subplot(1, 3, 2)
    plt.hist(contrast_responses, bins=bins)
    plt.title('Contrast Responses')
    plt.subplot(1, 3, 3)
    plt.scatter(contrast_responses, border_responses)
    plt.plot([0, 200], [0, 200], 'k')
    plt.xlabel('Contrast Responses')
    plt.ylabel('Border Responses')
    plt.tight_layout()
    plt.savefig('border-ownership-{}.eps'.format(layer))
    plt.savefig('border-ownership-{}.jpg'.format(layer))
    # plt.show()


if __name__ == '__main__':
    # Load pre-trained models
    doc_converted = doc.KitModel('model/doc.npy')
    hed_converted = hed.KitModel('model/hed.npy')
            
#     parameters, responses, preferred_stimuli = find_optimal_bars(doc_converted, hed_converted)
#     with open('doc_hed/preferred-stimuli.pkl', 'wb') as file:
#         pickle.dump({'parameters': parameters, 'responses': responses, 'preferred_stimuli': preferred_stimuli}, file)

    with open('doc_hed/preferred-stimuli.pkl', 'rb') as file:
        data = pickle.load(file)
    parameters = data['parameters']
    responses = data['responses']
    preferred_stimuli = data['preferred_stimuli']

    standard_test(doc_converted, hed_converted, preferred_stimuli)

    # import csv
    # with open('foo.csv', 'w', newline='\r\n') as csvfile:
    #     spamwriter = csv.writer(csvfile, delimiter=',')
    #     spamwriter.writerow(('count', 'group'))
    #     spamwriter.writerow(('10', 'a'))
    #     spamwriter.writerow(('12', 'a'))
    #
    # # in R studio ...
    # # data <- read.csv(file=file.choose(), header=TRUE, sep=",")
    
