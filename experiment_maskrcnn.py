import os
import pickle
import glob
import numpy as np
import skimage.io as sio
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import matplotlib.pyplot as plt
from caffe2.python import workspace
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.core.config import assert_and_infer_cfg
import detectron.core.test_engine as infer_engine
from detectron.core.test import im_detect_mask, im_detect_bbox, box_results_with_nms_and_limit
from detectron.utils.io import cache_url
from detectron.utils.timer import Timer
from border.stimuli import Colours, get_image, add_rectangle
import detectron.utils.c2 as c2_utils

c2_utils.import_detectron_ops()
cv2.ocl.setUseOpenCL(False)


def make_bar_stimuli(directory='.'):
    """
    Creates and saves bar stimulus images. These are used for finding optimal bar
    stimuli for units in a CNN, approximating the procedure in:

    H. Zhou, H. S. Friedman, and R. von der Heydt, "Coding of border ownership in monkey visual
    cortex.," J. Neurosci., vol. 20, no. 17, pp. 6594-6611, 2000.
    """

    colours = Colours()

    bg_colour_name = 'Light gray (background)'
    bg_colour = colours.get_RGB(bg_colour_name, 0)

    fg_colour_names = [key for key in colours.colours.keys()
                       if key != bg_colour_name]

    # TODO: probably need more sizes and angles
    lengths = [40, 80]
    widths = [4, 8]
    angles = np.pi * np.array([0, .25, .5, .75])

    parameters = []

    for fg_colour_name in fg_colour_names:
        n_luminances = colours.get_num_luminances(fg_colour_name)
        n_stimuli = len(lengths) * len(widths) * len(angles) * n_luminances
        print('Creating {} {} stimuli'.format(n_stimuli, fg_colour_name))
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

                        filename = 'bar{}.jpg'.format(len(parameters)-1)
                        sio.imsave(os.path.join(directory, filename), stimulus)

    return parameters


def get_model():
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    merge_cfg_from_file('/home/bryan/code/detectron/configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml')
    cfg.NUM_GPUS = 1
    assert_and_infer_cfg(cache_urls=False)
    DOWNLOAD_CACHE = '/tmp/detectron-download-cache'
    weights_url = 'https://s3-us-west-2.amazonaws.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl'
    weights = cache_url(weights_url, DOWNLOAD_CACHE)
    return infer_engine.initialize_model_from_cfg(weights)


def run_mask_net(model, image, layer, unit_index):
    with c2_utils.NamedCudaScope(0):
        box_proposals = None
        scores, boxes, im_scale = im_detect_bbox(
            model, image, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, boxes=box_proposals
        )
        boxes = np.array([[0, 0, 399, 399]]) # we don't want box tightly wrapped around stimulus
        im_detect_mask(model, im_scale, boxes)
        activities = workspace.blobs['gpu_0/{}'.format(layer)]
        centre = (int(activities.shape[2] / 2), int(activities.shape[3] / 2))
        response = activities[0, unit_index, centre[0], centre[1]]
        return response


def find_optimal_bars(image_directory, layer):
    model = get_model()
    im_list = glob.iglob(image_directory + '/*.jpg')
    im_list = [x for x in im_list] # so we can get length

    max_sd_per_map = [None] * len(im_list)
    for i, im_name in enumerate(im_list):
        print('processing {}'.format(im_name))
        im = cv2.imread(im_name)

        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                model, im, None, timers=None
            )
            responses = workspace.blobs['gpu_0/{}'.format(layer)]
            sd = []
            for i in range(responses.shape[0]):
                sd.append(np.std(responses[i,:,:,:]))
            max_sd_across_boxes = responses[np.argmax(sd),:,:,:]

            # stimuli aren't processed in order, so get index from file name
            bar_name = os.path.basename(im_name)[3:-4]
            bar_num = int(bar_name)
            max_sd_per_map[bar_num] = np.max(np.max(max_sd_across_boxes, axis=-1), axis=-1)

    max_sd_per_map = np.array(max_sd_per_map)
    result = np.argmax(max_sd_per_map, axis=0)
    print result   # TODO: check this
    return result



def standard_test(model, layer, unit_index, preferred_stimulus):
    colours = Colours()
    bg_colour_name = 'Light gray (background)'
    bg_colour = colours.get_RGB(bg_colour_name, 0)

    preferred_colour = preferred_stimulus['colour']

    square_shape = (100,100)

    angle = preferred_stimulus['angle']
    rotation = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    position_1 = np.add(np.dot(rotation, np.array([-50, 0]).transpose()), [200,200]).astype(np.int)
    position_2 = np.add(np.dot(rotation, np.array([50, 0]).transpose()), [200,200]).astype(np.int)

    # Stimuli as in panels A-D of Zhou et al. Figure 2
    stimulus_A = get_image((400, 400, 3), preferred_colour)
    add_rectangle(stimulus_A, position_1, square_shape, angle, bg_colour)

    stimulus_B = get_image((400, 400, 3), bg_colour)
    add_rectangle(stimulus_B, position_2, square_shape, angle, preferred_colour)

    stimulus_C = get_image((400, 400, 3), bg_colour)
    add_rectangle(stimulus_C, position_1, square_shape, angle, preferred_colour)

    stimulus_D = get_image((400, 400, 3), preferred_colour)
    add_rectangle(stimulus_D, position_2, square_shape, angle, bg_colour)

    stimulus_pref = get_image((400, 400, 3), bg_colour)
    add_rectangle(stimulus_pref,
                  [200,200],
                  (preferred_stimulus['width'], preferred_stimulus['length']),
                  preferred_stimulus['angle'],
                  preferred_colour)

    A = run_mask_net(model, stimulus_A, layer, unit_index)
    B = run_mask_net(model, stimulus_B, layer, unit_index)
    C = run_mask_net(model, stimulus_C, layer, unit_index)
    D = run_mask_net(model, stimulus_D, layer, unit_index)
    responses = [A, B, C, D]
    m = np.mean(responses)
    side = np.abs((A+C)/2 - (B+D)/2) / m * 100
    contrast = np.abs((A+B)/2 - (C+D)/2) / m * 100
    print('side: {} contrast: {}'.format(side, contrast))

    return {'responses': responses, 'side': side, 'contrast': contrast, 'mean': m}


def standard_test_full_layer(parameters, preferred_stimuli, layer, base_path='.'):
    m = len(preferred_stimuli)
    model = get_model()

    border_responses = []
    contrast_responses = []
    means = []
    responses = []
    for unit_index in range(m):
        print('{} of {} for {}'.format(unit_index, m, layer))
        result = standard_test(model, layer, unit_index, parameters[preferred_stimuli[unit_index]])
        border_responses.append(result['side'])
        contrast_responses.append(result['contrast'])
        means.append(result['mean'])
        responses.append(result['responses'])
        print(result['responses'])

    with open(os.path.join(base_path, 'border-rcnn-{}.pkl'.format(layer)), 'wb') as file:
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
    plt.savefig('border-ownership-{}.eps'.format(layer))
    plt.savefig('border-ownership-{}.jpg'.format(layer))


if __name__ == '__main__':
    MAKE_BARS = False
    FIND_PREFERRED_BARS = False
    DO_STANDARD_TEST = True
    # layer = 'conv5_mask'
    # layer = 'mask_fcn_logits'
    # layer = '_[mask]_fcn1'
    # layer = '_[mask]_fcn2'
    layer = '_[mask]_fcn3'
    # layer = '_[mask]_fcn4'
    # layer = 'mask_fcn_probs'

    stimulus_directory = '/home/bryan/code/DOC-tf/generated-files/bars'

    if MAKE_BARS:
        parameters = make_bar_stimuli(directory=stimulus_directory)
        with open(os.path.join(stimulus_directory, 'parameters.pkl'), 'wb') as f:
            pickle.dump(parameters, f)
    else:
        with open(os.path.join(stimulus_directory, 'parameters.pkl'), 'rb') as f:
            parameters = pickle.load(f)

    if FIND_PREFERRED_BARS:
        preferred_stimuli = find_optimal_bars(stimulus_directory, layer)
        with open('preferred_stimuli-{}'.format(layer), 'wb') as f:
            pickle.dump(preferred_stimuli, f)
    else:
        with open('preferred_stimuli-{}'.format(layer), 'rb') as f:
            preferred_stimuli = pickle.load(f)

    if DO_STANDARD_TEST:
        standard_test_full_layer(parameters, preferred_stimuli, layer)

