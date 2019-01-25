import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

base_path = './doc_size/150x150/'
layers = ['relu1_1', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3']

for i, layer in enumerate(layers):

    # Load data
    with open(os.path.join(base_path, 'border-'+layer+':0.pkl'), 'rb') as file:
        data = pickle.load(file)

    border_responses = data['border_responses']
    border_responses2 = data['border_responses2']

    # Plot BOS for different size squares
    inds = (~np.isnan(border_responses)) & (~np.isnan(border_responses2))
    border_responses = np.array(border_responses)[inds]
    border_responses2 = np.array(border_responses2)[inds]

    # from Bryan's plots code
    bins = np.linspace(0, 200, 21)

    nullfmt = NullFormatter()

    # definitions for the axes
    left, width = 0.12, 0.56
    bottom, height = 0.12, 0.56
    bottom_h = left_h = left + width + 0.11

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.13]
    rect_histy = [left_h, bottom, 0.13, height]

    # start with a rectangular Figure
    plt.figure(1, figsize=(9, 9))

    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    axScatter.scatter(border_responses, border_responses2, c='k')

    axScatter.set_xticks([0, 100, 200])
    axScatter.set_yticks([0, 100, 200])
    axScatter.tick_params(axis='both', labelsize=24)

    axHistx.hist(border_responses, bins=bins, color='k')
    axHistx.tick_params(axis='both', labelsize=24)
    axHisty.hist(border_responses2, bins=bins,
                 orientation='horizontal', color='k')
    axHisty.tick_params(axis='both', labelsize=24)

    axHistx.set_xticks([0, 100, 200])
    axHisty.set_yticks([0, 100, 200])
    axHistx.set_title(layer, fontsize=30, y=1.2)

    # if i == 0:
    #     axScatter.set_ylabel(
    #         'Border preference score \n(large square)', fontsize=28)
    # if i == 2:
    #     axScatter.set_xlabel(
    #         'Border preference score \n(standard square)', fontsize=28)

    plt.savefig(os.path.join(base_path, '{}_size_tuning.eps'.format(layer)))
    # plt.show()

# Close all plots
plt.close()
