import copy
import os
import pickle
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from border.stimuli import Colours, get_image, add_rectangle


def plot_border_and_contrast(result_dir, error_bars=True):
    files = [x for x in os.listdir(result_dir) if 'border' in x and 'pkl' in x]

    border_means = []
    contrast_means = []
    difference_means = []
    difference_sds = []
    for file in files:
        with open(os.path.join(result_dir, file), 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            border_means.append(np.nanmean(data['border_responses']))
            contrast_means.append(np.nanmean(data['contrast_responses']))
            difference = np.array(data['border_responses']) - np.array(data['contrast_responses'])
            difference_means.append(np.nanmean(difference))
            difference_sds.append(np.nanstd(difference))

    layers = np.arange(1, len(border_means)+1)
    if error_bars:
        plt.errorbar(layers, difference_means, yerr=difference_sds, fmt='-o', c='k', capsize=3)
    else:
        plt.plot(layers, difference_means, 'k--')


def _plot_border_vs_contrast(border_responses, contrast_responses, dir, layer):
    #adapted from https://matplotlib.org/examples/pylab_examples/scatter_hist.html
    bins = np.linspace(0, 200, 21)

    nullfmt = NullFormatter()

    # definitions for the axes
    left, width = 0.17, 0.56
    bottom, height = 0.17, 0.56
    bottom_h = left_h = left + width + 0.11

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.13]
    rect_histy = [left_h, bottom, 0.13, height]

    # start with a rectangular Figure
    plt.figure(1, figsize=(2.2, 2.2))

    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    axScatter.scatter(border_responses, contrast_responses, c='k')

    axScatter.set_xlim((0, 200))
    axScatter.set_ylim((0, 200))

    axHistx.hist(border_responses, bins=bins, color='k')
    axHisty.hist(contrast_responses, bins=bins, orientation='horizontal', color='k')

    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())
    plt.savefig(os.path.join(dir, '{}.eps'.format(layer)))


def plot_border_vs_contrast(result_dir, layers, add_suffix=True):
    if add_suffix:
        files = ['border-{}:0.pkl'.format(layer) for layer in layers]
    else:
        files = ['border-{}.pkl'.format(layer) for layer in layers]

    for layer, file in zip(layers, files):
        with open(os.path.join(result_dir, file), 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            border_responses = data['border_responses']
            contrast_responses = data['contrast_responses']

            r = np.array(data['responses'])
            plt.plot(r)
            plt.show()

            border_responses = [br for br in border_responses if not np.isnan(br)]
            contrast_responses = [cr for cr in contrast_responses if not np.isnan(cr)]
            _plot_border_vs_contrast(border_responses, contrast_responses, result_dir, layer)


def find_example_cells(result_dir, layer):
    file = 'border-{}:0.pkl'.format(layer)

    with open(os.path.join(result_dir, file), 'rb') as f:
        data = pickle.load(f)
        border_responses = data['border_responses']
        contrast_responses = data['contrast_responses']

        # plt.plot(border_responses, 'o-')
        # plt.plot(contrast_responses, 'o-')
        # plt.legend(('border', 'contrast'))

        diff = np.array(border_responses) - np.array(contrast_responses)
        print(np.argwhere(diff > 150))
        plt.plot(diff, 'o')
        plt.ylabel('border minus contrast')
        plt.xlabel('feature map #')
        plt.show()


def anova_results(file):
    threshold = .01
    obj = 0
    fore = 0
    both = 0
    none = 0

    with open(file) as f:
        reader = csv.reader(f)
        for row in reader:
            # print(row)
            if row[0] and not row[2] == 'NA':
                po = float(row[2])
                pf = float(row[3])

                if po < threshold and pf < threshold:
                    both = both + 1
                elif po < threshold:
                    obj = obj + 1
                elif pf < threshold:
                    fore = fore + 1
                else:
                    none = none + 1

    total = both + obj + fore + none
    print('both {} o {} f {} none {}'.format(both/total, obj/total, fore/total, none/total))


def get_stimulus_A(preferred_stimulus, im_width=400):
    colours = Colours()
    bg_colour_name = 'Light gray (background)'
    bg_colour = colours.get_RGB(bg_colour_name, 0)

    preferred_colour = preferred_stimulus['colour']

    square_shape = (im_width / 4, im_width / 4)

    angle = preferred_stimulus['angle']
    rotation = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    offset = im_width / 8
    centre = im_width / 2
    position_1 = np.add(np.dot(rotation, np.array([-offset, 0]).transpose()), [centre, centre]).astype(np.int)
    position_2 = np.add(np.dot(rotation, np.array([offset, 0]).transpose()), [centre, centre]).astype(np.int)

    stimulus_A = get_image((im_width, im_width, 3), preferred_colour)
    add_rectangle(stimulus_A, position_1, square_shape, angle, bg_colour)
    return stimulus_A


def get_stimulus_bounds(preferred_stimulus, im_width=400):
    stim_width = im_width / 4
    angle = preferred_stimulus['angle']
    rotation = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    offset = im_width / 8 * np.vstack((np.ones((1,4)), np.zeros((1,4))))
    centre = im_width / 2
    square_corners = np.array([[-stim_width/2, stim_width/2, stim_width/2, -stim_width/2],
                               [-stim_width/2, -stim_width/2, stim_width/2, stim_width/2]])
    corners1 = np.dot(rotation, square_corners+offset) + centre
    corners2 = np.dot(rotation, square_corners-offset) + centre
    return corners1, corners2


def draw_square(corners):
    first_corner = (corners[:,0][None]).T
    corners = np.append(corners, first_corner, axis=1)
    plt.plot(corners[0,:], corners[1,:], 'k', linewidth=1)


def inner_product(stimulus1, stimulus2, mask=None):
    """
    Meant for finding integral of optimal stimulus product with colour over a square.
    """
    def apply_mask(image):
        result = copy.copy(image)
        for i in range(image.shape[2]):
            result[:,:,i] = result[:,:,i] * mask
        return result

    if mask is None:
        return np.dot(stimulus1.flatten(), stimulus2.flatten())
    else:
        return np.dot(apply_mask(stimulus1).flatten(), apply_mask(stimulus2).flatten())


def get_mask(preferred_stimulus, im_width, include_left, include_right, include_background):
    square_shape = (im_width / 4, im_width / 4)
    angle = preferred_stimulus['angle']
    rotation = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    offset = im_width / 8
    centre = im_width / 2
    position_1 = np.add(np.dot(rotation, np.array([-offset, 0]).transpose()), [centre, centre]).astype(np.int)
    position_2 = np.add(np.dot(rotation, np.array([offset, 0]).transpose()), [centre, centre]).astype(np.int)

    stimulus_left = get_image((im_width, im_width, 3), (0,0,0))
    add_rectangle(stimulus_left, position_1, square_shape, angle, (1,1,1))
    mask_left = stimulus_left[:,:,0] > .5

    stimulus_right = get_image((im_width, im_width, 3), (0,0,0))
    add_rectangle(stimulus_right, position_2, square_shape, angle, (1,1,1))
    mask_right = stimulus_right[:,:,0] > .5

    mask_both = np.logical_or(mask_left, mask_right)
    mask_bg = np.full((im_width, im_width), True)
    mask_bg = np.logical_xor(mask_bg, mask_both)

    result = np.full((im_width, im_width), False)

    if include_left:
        result = np.logical_or(result, mask_left)

    if include_right:
        result = np.logical_or(result, mask_right)

    if include_background:
        result = np.logical_or(result, mask_bg)

    return result


def inner_product_differences(optimal_stimulus, preferred_stimulus, im_width):
    colours = Colours()
    bg_colour_name = 'Light gray (background)'
    bg_colour = colours.get_RGB(bg_colour_name, 0)
    pref_colour = preferred_stimulus['colour']

    # unrealistically large negative effect of surround if we don't centre this
    optimal_stimulus = optimal_stimulus - np.mean(optimal_stimulus)

    bg_image = get_image((im_width, im_width, 3), bg_colour) - .5 # centering these too
    pref_image = get_image((im_width, im_width, 3), pref_colour) - .5

    pref_left = inner_product(optimal_stimulus, pref_image, mask=get_mask(preferred_stimulus, im_width, True, False, False))
    pref_right = inner_product(optimal_stimulus, pref_image, mask=get_mask(preferred_stimulus, im_width, False, True, False))
    pref_surround = inner_product(optimal_stimulus, pref_image, mask=get_mask(preferred_stimulus, im_width, False, False, True))
    bg_left = inner_product(optimal_stimulus, bg_image, mask=get_mask(preferred_stimulus, im_width, True, False, False))
    bg_right = inner_product(optimal_stimulus, bg_image, mask=get_mask(preferred_stimulus, im_width, False, True, False))
    bg_surround = inner_product(optimal_stimulus, bg_image, mask=get_mask(preferred_stimulus, im_width, False, False, True))

    side = (pref_left + bg_right) - (pref_right + bg_left)
    surround = pref_surround - bg_surround

    # print('pref {} {} {}'.format(pref_left, pref_right, pref_surround))
    # print('bg {} {} {}'.format(bg_left, bg_right, bg_surround))
    # print('side {} surround {}'.format(side, surround))

    return side, surround


def get_preferred_stimulus(index):
    # for DOC relu5_3
    with open('./doc/preferred-stimuli.pkl', 'rb') as f:
        data = pickle.load(f)
        pref_index = data['preferred_stimuli']['relu5_3:0'][index]
        return data['parameters'][pref_index]


def plot_stimulus_A(index):
    # for DOC relu5_3
    with open('./doc/preferred-stimuli.pkl', 'rb') as f:
        data = pickle.load(f)
        pref_index = data['preferred_stimuli']['relu5_3:0'][index]
        pref_stimulus = data['parameters'][pref_index]
        A = get_stimulus_A(pref_stimulus)
        plt.imshow(A)


def get_stimuli(preferred_stimulus, im_width):
    # TODO: extract this method in experiment

    colours = Colours()
    bg_colour_name = 'Light gray (background)'
    bg_colour = colours.get_RGB(bg_colour_name, 0)

    preferred_colour = preferred_stimulus['colour']

    square_shape = (im_width/4, im_width/4)

    angle = preferred_stimulus['angle']
    rotation = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    offset = im_width/8
    centre = im_width/2
    position_1 = np.add(np.dot(rotation, np.array([-offset, 0]).transpose()), [centre,centre]).astype(np.int)
    position_2 = np.add(np.dot(rotation, np.array([offset, 0]).transpose()), [centre,centre]).astype(np.int)

    stimulus_A = get_image((im_width, im_width, 3), preferred_colour)
    add_rectangle(stimulus_A, position_1, square_shape, angle, bg_colour)

    stimulus_B = get_image((im_width, im_width, 3), bg_colour)
    add_rectangle(stimulus_B, position_2, square_shape, angle, preferred_colour)

    stimulus_C = get_image((im_width, im_width, 3), bg_colour)
    add_rectangle(stimulus_C, position_1, square_shape, angle, preferred_colour)

    stimulus_D = get_image((im_width, im_width, 3), preferred_colour)
    add_rectangle(stimulus_D, position_2, square_shape, angle, bg_colour)

    return stimulus_A, stimulus_B, stimulus_C, stimulus_D


def get_optimal_stimulus(index, border=True):
    if index in (121, 204, 254, 326, 476): # these had strongest border coding bias
        filename = './visualize/border/visualize-convolution_12-{}-8.pkl'.format(index)
    else:
        filename = './visualize/contrast/visualize-convolution_12-{}-8.pkl'.format(index)

    with open(filename, 'rb') as f:
        return pickle.load(f)


def inner_product_experiment(index):
    optimal_stimulus = get_optimal_stimulus(index)
    preferred_stimulus = get_preferred_stimulus(index)
    im_width = optimal_stimulus.shape[0]
    stimulus_A, stimulus_B, stimulus_C, stimulus_D = get_stimuli(preferred_stimulus, im_width)

    # centering everything
    optimal_stimulus = optimal_stimulus - np.mean(optimal_stimulus)

    A = max(inner_product(optimal_stimulus, stimulus_A-.5), 0)
    B = max(inner_product(optimal_stimulus, stimulus_B-.5), 0)
    C = max(inner_product(optimal_stimulus, stimulus_C-.5), 0)
    D = max(inner_product(optimal_stimulus, stimulus_D-.5), 0)
    m = np.mean([A, B, C, D])
    side = np.abs((A+C)/2 - (B+D)/2) / m * 100
    contrast = np.abs((A+B)/2 - (C+D)/2) / m * 100
    return side, contrast


if __name__ == '__main__':
    """
    DOC relu5_3: 
    Good border cells: 
    [[121]
     [204]
     [254]
     [326]
     [476]]
    Good contrast cells:
    [[ 81]
     [ 94]
     [199]
     [205]
     [226]
     [328]
     [491]]
    """

    # index = 81
    # preferred_stimulus = get_preferred_stimulus(index)
    # mask = get_mask(preferred_stimulus, 256, False, False, True)
    # plt.imshow(mask)
    # plt.show()

    border_units = (121, 204, 254, 326, 476)
    contrast_units = (81, 94, 199, 205, 226, 328, 491)

    # def get_differences(units):
    #     sides = []
    #     surrounds = []
    #     for index in units:
    #         optimal_stimulus = get_optimal_stimulus(index)
    #         preferred_stimulus = get_preferred_stimulus(index)
    #         im_width = optimal_stimulus.shape[0]
    #         side, surround = inner_product_differences(optimal_stimulus, preferred_stimulus, im_width)
    #         sides.append(side)
    #         surrounds.append(surround)
    #     return sides, surrounds
    #
    # border_sides, border_surrounds = get_differences(border_units)
    # print('{} +/- {}'.format(np.mean(border_sides), np.std(border_sides)))
    # print('{} +/- {}'.format(np.mean(border_surrounds), np.std(border_surrounds)))
    #
    # contrast_sides, contrast_surrounds = get_differences(contrast_units)
    # print('{} +/- {}'.format(np.mean(contrast_sides), np.std(contrast_sides)))
    # print('{} +/- {}'.format(np.mean(contrast_surrounds), np.std(contrast_surrounds)))

    border_sides = []
    border_contrasts = []
    for index in border_units:
        side, contrast = inner_product_experiment(index)
        border_sides.append(side)
        border_contrasts.append(contrast)
    print('Border cells:')
    print('Border {} +/- {}'.format(np.mean(border_sides), np.std(border_sides)))
    print('Contrast {} +/- {}'.format(np.mean(border_contrasts), np.std(border_contrasts)))
    print(border_sides)
    print(border_contrasts)

    contrast_sides = []
    contrast_contrasts = []
    for index in contrast_units:
        side, contrast = inner_product_experiment(index)
        contrast_sides.append(side)
        contrast_contrasts.append(contrast)
    print('Contrast cells:')
    print('Border {} +/- {}'.format(np.mean(contrast_sides), np.std(contrast_sides)))
    print('Contrast {} +/- {}'.format(np.mean(contrast_contrasts), np.std(contrast_contrasts)))
    print(contrast_sides)
    print(contrast_contrasts)

    # plt.figure(figsize=(6,3))
    # plt.subplot(1,2,1)
    # index = border_units[0]
    # optimal_stimulus = get_optimal_stimulus(index)
    # preferred_stimulus = get_preferred_stimulus(index)
    # corners1, corners2 = get_stimulus_bounds(preferred_stimulus, im_width=256)
    # plt.imshow(optimal_stimulus)
    # draw_square(corners1)
    # draw_square(corners2)
    # plt.xticks([]), plt.yticks([])
    # plt.title('Strong Border Response')
    # plt.subplot(1,2,2)
    # index = contrast_units[0]
    # optimal_stimulus = get_optimal_stimulus(index)
    # preferred_stimulus = get_preferred_stimulus(index)
    # corners1, corners2 = get_stimulus_bounds(preferred_stimulus, im_width=256)
    # plt.imshow(optimal_stimulus)
    # draw_square(corners1)
    # draw_square(corners2)
    # plt.xticks([]), plt.yticks([])
    # plt.title('Strong Contrast Response')
    # plt.tight_layout()
    # plt.savefig('optimal-stim-examples.eps')
    # plt.show()


    # index = 81
    # plt.subplot(1,2,1)
    # plot_stimulus_A(index)
    # plt.subplot(1,2,2)
    # optimal_stimulus = get_optimal_stimulus(index)
    # plt.imshow(optimal_stimulus)
    # # plot_stimulus_optimal(index)
    # preferred_stimulus = get_preferred_stimulus(index)
    # corners1, corners2 = get_stimulus_bounds(preferred_stimulus, im_width=256)
    # draw_square(corners1)
    # draw_square(corners2)
    # inner_product_differences(optimal_stimulus, preferred_stimulus, optimal_stimulus.shape[0])
    # # inner_product(None, None, preferred_stimulus['colour'])
    # plt.show()

    #TODO: calculate inner product of positive and negative object and surround regions
    # with optimal stimulus


    # layers = ['relu1_1', 'relu2_2',
    #           'relu3_3', 'relu4_3', 'relu5_3']
    # plot_border_vs_contrast('./doc', layers)

    # layers = ['mask_fcn_probs']
    # layers = ['_[mask]_fcn4']
    # plot_border_vs_contrast('./generated-files/mask-rcnn', layers, add_suffix=False)
    # plt.show()

    # anova_results('./generated-files/doc/border-relu1_1_0/probabilities.csv')
    # anova_results('./generated-files/doc/border-relu5_3_0/probabilities.csv')

    # plt.figure(figsize=(6,2.5))
    # plt.subplot(1,2,1)
    # plot_border_and_contrast('./hed')
    # plot_border_and_contrast('./generated-files/small-square-hed', error_bars=False)
    # plt.title('Boundary Branch')
    # plt.xlabel('Nonlinear Layer #')
    # plt.ylabel('Border minus Contrast')
    # plt.ylim((-200,200))
    # plt.subplot(1,2,2)
    # plot_border_and_contrast('./doc')
    # plot_border_and_contrast('./generated-files/small-square-doc', error_bars=False)
    # plt.title('Orientation Branch')
    # plt.xlabel('Nonlinear Layer #')
    # plt.ylim((-200,200))
    # plt.tight_layout()
    # plt.savefig('DOC-border-minus-contrast.eps')
    # plt.show()

    # plt.figure(figsize=(6,2.5))
    # plt.subplot(1,7,(1,5))
    # plot_border_and_contrast('./generated-files/resnet')
    # plt.title('ResNet')
    # plt.xlabel('Nonlinear Layer #')
    # plt.ylabel('Border minus Contrast')
    # plt.ylim((-200,200))
    # plt.subplot(1,7,(6,7))
    # plot_border_and_contrast('./generated-files/mask-rcnn')
    # plt.title('Mask R-CNN')
    # plt.xlabel('Nonlinear Layer #')
    # plt.ylim((-200,200))
    # plt.xticks([2,5])
    # plt.tight_layout()
    # plt.savefig('others-border-minus-contrast.eps')
    # plt.show()

    # plot_border_and_contrast('./generated-files/resnet')
    # plot_border_and_contrast('./generated-files/mask-rcnn')
    # plt.show()

    # find_example_cells('./hed', 'relu5_3')
