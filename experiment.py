import csv
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from border.stimuli import Colours, get_image, add_rectangle
from keras import applications


def find_optimal_bars(input, layers, im_shape=(400,400)):
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

                            stimulus = get_image((im_shape[0], im_shape[1], 3), bg_colour)
                            add_rectangle(stimulus, (im_shape[0]/2, im_shape[1]/2),
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


def standard_test(input, layer, unit_index, preferred_stimulus, im_width):
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

    preferred_colour = preferred_stimulus['colour']

    # square_shape = (im_width/4, im_width/4)
    square_shape = (im_width/8, im_width/8) ###########

    angle = preferred_stimulus['angle']
    rotation = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    # offset = im_width/8
    offset = im_width/16 ##################
    centre = im_width/2
    position_1 = np.add(np.dot(rotation, np.array([-offset, 0]).transpose()), [centre,centre]).astype(np.int)
    position_2 = np.add(np.dot(rotation, np.array([offset, 0]).transpose()), [centre,centre]).astype(np.int)

    # preferred_shape = (preferred_stimulus['width'], preferred_stimulus['length'])
    # add_rectangle(stimulus, (200,200), preferred_shape, angle, preferred_colour)

    # Stimuli as in panels A-D of Zhou et al. Figure 2
    stimulus_A = get_image((im_width, im_width, 3), preferred_colour)
    add_rectangle(stimulus_A, position_1, square_shape, angle, bg_colour)

    stimulus_B = get_image((im_width, im_width, 3), bg_colour)
    add_rectangle(stimulus_B, position_2, square_shape, angle, preferred_colour)

    stimulus_C = get_image((im_width, im_width, 3), bg_colour)
    add_rectangle(stimulus_C, position_1, square_shape, angle, preferred_colour)

    stimulus_D = get_image((im_width, im_width, 3), preferred_colour)
    add_rectangle(stimulus_D, position_2, square_shape, angle, bg_colour)

    stimulus_pref = get_image((im_width, im_width, 3), bg_colour)
    add_rectangle(stimulus_pref,
                  [centre,centre],
                  (preferred_stimulus['width'], preferred_stimulus['length']),
                  preferred_stimulus['angle'],
                  preferred_colour)

    input_data = np.stack((stimulus_A, stimulus_B, stimulus_C, stimulus_D, stimulus_pref))

    # print(input_data.shape)
    # plt.imshow(stimulus_D)
    # plt.show()

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        model_tf = sess.graph.get_tensor_by_name(layer)
        activities = sess.run(
            model_tf, feed_dict={input: input_data})

        centre = (int(activities.shape[1] / 2), int(activities.shape[2] / 2))
        responses = activities[:, centre[0], centre[1], unit_index]

    m = np.mean(responses[:4])
    A, B, C, D, P = responses
    side = np.abs((A+C)/2 - (B+D)/2) / m * 100
    contrast = np.abs((A+B)/2 - (C+D)/2) / m * 100
    # print('side: {} contrast: {}'.format(side, contrast))

    return {'responses': responses, 'side': side, 'contrast': contrast, 'mean': m}

    #TODO: the side does involve a contrast difference if the square doesn't fully cover the classical RF,
    #   but as this net is feedforward, there can't be a border effect if it does


def count_feature_maps(layer):
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        model_tf = sess.graph.get_tensor_by_name(layer)
        result = model_tf.shape[3]

    return result


def clean_layer_name(layer):
    return layer.replace(':', '_').replace('/', '_')


def standard_test_full_layer(layer, preferred_stimuli, im_width=400, base_path='.'):
    m = count_feature_maps(layer)

    border_responses = []
    contrast_responses = []
    means = []
    responses = []
    for unit_index in range(m):
        print('{} of {} for {}'.format(unit_index, m, layer))
        result = standard_test(input_tf, layer, unit_index, parameters[preferred_stimuli[layer][unit_index]], im_width=im_width)
        border_responses.append(result['side'])
        contrast_responses.append(result['contrast'])
        means.append(result['mean'])
        responses.append(result['responses'])
        print(result['responses'])

    filename = 'border-{}.pkl'.format(clean_layer_name(layer))
    with open(os.path.join(base_path, filename), 'wb') as file:
        pickle.dump({'border_responses': border_responses,
                     'contrast_responses': contrast_responses,
                     'means': means,
                     'responses': responses}, file)

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
    plt.savefig(os.path.join(base_path, 'border-ownership-{}.eps'.format(clean_layer_name(layer))))
    plt.savefig(os.path.join(base_path, 'border-ownership-{}.jpg'.format(clean_layer_name(layer))))
    # plt.show()


def get_poisson_folder(base_folder, layer):
    """
    This is used in a couple of different places, so it's extracted here to ensure
    consistency.

    :param base_folder: folder that contains stardard_test results
    :param layer: network layer name
    :return: sub-folder for Poisson model results
    """
    return '{}/border-{}'.format(base_folder, layer.replace(':', '_'))


def export_poisson(base_folder, layer):
    """
    Uses data saved during standard_test. Exports fake Poisson spike counts
    to CSV. The intent is to use R for ANOVA, for comparison with Zhou et al.
    Figure 16.

    :param base_folder: folder that contains stardard_test results
    :param layer: layer name
    """

    max_rate = 50
    trial_duration = 1  # " ... based on mean firing rates during successive 1 sec intervals"

    with open('{}/border-{}.pkl'.format(base_folder, layer), 'rb') as file:
        data = pickle.load(file)

    destination_folder = get_poisson_folder(base_folder, layer)
    os.makedirs(destination_folder, exist_ok=True)

    for i in range(len(data['responses'])):
        A, B, C, D, P = data['responses'][i]
        result_file = '{}/poisson-{:04}.csv'.format(destination_folder, i)

        if P > 0:
            with open(result_file, 'w', newline='\r\n') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                writer.writerow(('count', 'condition', 'object', 'foreground'))

                mean_count_A = A / P * max_rate * trial_duration
                mean_count_B = B / P * max_rate * trial_duration
                mean_count_C = C / P * max_rate * trial_duration
                mean_count_D = D / P * max_rate * trial_duration

                for j in range(10):
                    writer.writerow((np.random.poisson(mean_count_A), 'A', 'left', 'right'))
                    writer.writerow((np.random.poisson(mean_count_B), 'B', 'right', 'right'))
                    writer.writerow((np.random.poisson(mean_count_C), 'C', 'left', 'left'))
                    writer.writerow((np.random.poisson(mean_count_D), 'D', 'right', 'left'))


def plot_poisson(base_folder, layer):
    with open('{}/border-{}.pkl'.format(base_folder, layer), 'rb') as file:
        data = pickle.load(file)
    responses = data['responses']

    prob_file= os.path.join(get_poisson_folder(base_folder, layer), 'probabilities.csv')

    p_object = np.ones(len(responses))
    p_foreground = np.ones(len(responses))
    with open(prob_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0] == '':
                assert row[1] == 'index'
                assert row[2] == 'p-object'
                assert row[3] == 'p-foreground'
            else:
                index = int(row[1])
                try:
                    p_object[index] = float(row[2])
                    p_foreground[index] = float(row[3])
                except ValueError:
                    p_object[index] = None
                    p_foreground[index] = None


    side = np.array([np.nan] * len(responses))
    contrast = np.array([np.nan] * len(responses))
    for i in range(len(responses)):
        A, B, C, D, P = responses[i]
        if P > 0:
            side[i] = np.abs((A+C)/2 - (B+D)/2) / P
            contrast[i] = np.abs((A+B)/2 - (C+D)/2) / P

    significance_threshold = .01
    o_sig = p_object < significance_threshold
    f_sig = p_foreground < significance_threshold
    o_nsig = p_object > significance_threshold
    f_nsig = p_foreground > significance_threshold

    plt.scatter(side[np.logical_and(o_sig, f_nsig)], contrast[np.logical_and(o_sig, f_nsig)], c='r', marker='.')
    plt.scatter(side[np.logical_and(o_nsig, f_sig)], contrast[np.logical_and(o_nsig, f_sig)], c='g', marker='x')
    plt.scatter(side[np.logical_and(o_nsig, f_nsig)], contrast[np.logical_and(o_nsig, f_nsig)], c='g', marker='.')
    plt.scatter(side[np.logical_and(o_sig, f_sig)], contrast[np.logical_and(o_sig, f_sig)], c='r', marker='x')
    plt.xlabel('Normalized ownership difference')
    plt.xlabel('Normalized contrast difference')
    plt.legend(('only ownership', 'only contrast', 'neither', 'both'))
    plt.title('significance at alpha = {}'.format(significance_threshold))
    fig_file = os.path.join(get_poisson_folder(base_folder, layer), 'poisson-probabilities.eps')
    plt.savefig(fig_file)
    plt.show()


if __name__ == '__main__':
    # network = 'resnet'
    # network = 'doc'
    network = 'hed'
    FIND_OPTIMAL_BARS = False
    DO_STANDARD_TEST = True

    if network == 'resnet':
        im_width = 224
        input_tf = applications.ResNet50().input

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            ops = sess.graph.get_operations()
            layers = [op.name + ':0' for op in ops if 'Relu' in op.name]
            check = [sess.graph.get_tensor_by_name(layer) for layer in layers]
            # layers = layers[45:] ################# enter last complete layer number

    elif network == 'doc' or network == 'hed':
        im_width = 400
        if network == 'doc':
            from model.doc import KitModel
            model_converted = KitModel('model/doc.npy')
        else:
            from model.hed import KitModel
            model_converted = KitModel('model/hed.npy')

        input_tf, _ = model_converted

        # Define layers to perform experiment on
        layers = ['relu1_1', 'relu1_2', 'relu2_1', 'relu2_2', 'relu3_1', 'relu3_2',
                  'relu3_3', 'relu4_1', 'relu4_2', 'relu4_3', 'relu5_1', 'relu5_2', 'relu5_3']
        device_append = ':0'
        layers = [layer + device_append for layer in layers]

    else:
        raise Exception('unknown network')

    if FIND_OPTIMAL_BARS:
        parameters, responses, preferred_stimuli = find_optimal_bars(
            input_tf, layers, im_shape=(im_width, im_width))
        with open(network + '/preferred-stimuli.pkl', 'wb') as file:
            pickle.dump({'parameters': parameters, 'responses': responses,
                         'preferred_stimuli': preferred_stimuli}, file)
    else: # load from file
        with open(network + '/preferred-stimuli.pkl', 'rb') as file:
            data = pickle.load(file)
        parameters = data['parameters']
        responses = data['responses']
        preferred_stimuli = data['preferred_stimuli']

    if DO_STANDARD_TEST:
        for layer in layers:
            standard_test_full_layer(layer, preferred_stimuli, im_width=im_width,
                                     base_path='./generated-files/small-square-'+network)
        # export_poisson('./generated-files/'+network, layers[-1])
        # plot_poisson('./generated-files/'+network, layers[0])
