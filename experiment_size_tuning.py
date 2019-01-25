import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from border.stimuli import Colours, get_image, add_rectangle


def standard_test(input, layer, unit_index, preferred_stimulus):
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
    add_rectangle(stimulus_B, position_2, square_shape, angle, preferred_colour)

    stimulus_C = get_image((400, 400, 3), bg_colour)
    add_rectangle(stimulus_C, position_1, square_shape, angle, preferred_colour)

    stimulus_D = get_image((400, 400, 3), preferred_colour)
    add_rectangle(stimulus_D, position_2, square_shape, angle, bg_colour)
    
    # Stimulus of different size
    square_shape = (150, 150)
    
    stimulus_A2 = get_image((400, 400, 3), preferred_colour)
    add_rectangle(stimulus_A2, position_1, square_shape, angle, bg_colour)

    stimulus_B2 = get_image((400, 400, 3), bg_colour)
    add_rectangle(stimulus_B2, position_2, square_shape, angle, preferred_colour)

    stimulus_C2 = get_image((400, 400, 3), bg_colour)
    add_rectangle(stimulus_C2, position_1, square_shape, angle, preferred_colour)

    stimulus_D2 = get_image((400, 400, 3), preferred_colour)
    add_rectangle(stimulus_D2, position_2, square_shape, angle, bg_colour)

    input_data = np.stack((stimulus_A, stimulus_B, stimulus_C, stimulus_D,
                           stimulus_A2, stimulus_B2, stimulus_C2, stimulus_D2))

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
    m2 = np.mean(responses[4:])

    A, B, C, D, A2, B2, C2, D2 = responses
    side = np.abs((A+C)/2 - (B+D)/2) / m * 100
    side2 = np.abs((A2+C2)/2 - (B2+D2)/2) / m2 * 100

    return {'responses': responses, 'side': side, 'side2': side2, 'mean': m, 'mean2': m2}

    #TODO: the side does involve a contrast difference if the square doesn't fully cover the classical RF,
    #   but as this net is feedforward, there can't be a border effect if it does
    #TODO: generate 10 Poisson random samples from each condition mean for ANOVA
    #TODO: how to set scale for Poisson samples? maybe assume preferred stim evokes a certain rate?


def count_feature_maps(layer):
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        model_tf = sess.graph.get_tensor_by_name(layer)
        result = model_tf.shape[3]

    return result


def standard_test_full_layer(layer, preferred_stimuli):
    m = count_feature_maps(layer)

    border_responses = []
    border_responses2 = []
    means = []
    means2 = []
    for unit_index in range(m):
        print('{} of {} for {}'.format(unit_index, m, layer))
        result = standard_test(input_tf, layer, unit_index, parameters[preferred_stimuli[layer][unit_index]])
        border_responses.append(result['side'])
        border_responses2.append(result['side2'])
        means.append(result['mean'])
        means2.append(result['mean2'])

    with open('border-{}.pkl'.format(layer), 'wb') as file:
        pickle.dump({'border_responses': border_responses,
                     'border_responses2': border_responses2,
                     'means': means,
                     'means2': means2}, file)


if __name__ == '__main__':
    from model.doc import KitModel
    model_converted = KitModel('model/doc.npy')
    with open('doc/preferred-stimuli.pkl', 'rb') as file:
        data = pickle.load(file)
        
#     from model.hed import KitModel
#     model_converted = KitModel('model/hed.npy')
#     with open('hed/preferred-stimuli.pkl', 'rb') as file:
#         data = pickle.load(file)
    
    input_tf, _ = model_converted
    # Define layers to perform experiment on
    layers = ['relu1_1', 'relu1_2', 'relu2_1', 'relu2_2', 'relu3_1', 'relu3_2',
              'relu3_3', 'relu4_1', 'relu4_2', 'relu4_3', 'relu5_1', 'relu5_2', 'relu5_3']
    device_append = ':0'
    layers = [layer + device_append for layer in layers]

    parameters = data['parameters']
    responses = data['responses']
    preferred_stimuli = data['preferred_stimuli']

    for layer in layers:
        standard_test_full_layer(layer, preferred_stimuli)
