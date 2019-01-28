import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon
from skimage.color import xyz2rgb


class Colours:
    def __init__(self):
        """
        These are from Table 1 of H. Zhou, H. S. Friedman, and R. von der Heydt, “Coding of
        border ownership in monkey visual cortex.,” J. Neurosci., vol. 20, no. 17, pp. 6594–6611, 2000.
        """
        self.colours = {
            'Red–brown': {'x': 0.60, 'y': 0.35, 'Y': (14,2.7)},
            'Green-olive': {'x': 0.31, 'y': 0.58, 'Y': (37,6.7)},
            'Blue-azure': {'x': 0.16, 'y': 0.08, 'Y': (6.8,1.8)},
            'Yellow-beige': {'x': 0.41, 'y': 0.50, 'Y': (37,6.5)},
            'Violet-purple': {'x': 0.30, 'y': 0.15, 'Y': (20,3.4)},
            'Aqua-cyan': {'x': 0.23, 'y': 0.31, 'Y': (38,7.3)},
            'White-gray–black': {'x': 0.30, 'y': 0.32, 'Y': (38,8.8,1.2)},
            'Light gray (background)': {'x': 0.30, 'y': 0.32, 'Y': [20]}
        }
        self.black = [0, 0, 0]

    def _convert_xyY_XYZ(self, x, y, Y):
        # see https: // en.wikipedia.org / wiki / CIE_1931_color_space
        X = Y/y*x
        Z = Y/y*(1-x-y)
        return X, Y, Z

    def get_colour_names(self):
        """
        :return: Names of available stimulus hues
        """
        return self.colours.keys()

    def get_num_luminances(self, colour_name):
        """
        :param colour_name: Name of a stimulus hue
        :return: Number of available luminances for this hue
        """
        return len(self.colours[colour_name]['Y'])

    def get_RGB(self, colour_name, luminance_index):
        """
        :param colour_name: Name of a stimulus hue
        :param luminance_index: Index of a luminance value for this hue
        :return: Red-green-blue values of the specified hue and luminance
        """
        x = self.colours[colour_name]['x']
        y = self.colours[colour_name]['y']
        Y = self.colours[colour_name]['Y'][luminance_index]
        X, Y, Z = self._convert_xyY_XYZ(x, y, Y)
        XYZ = np.array((X,Y,Z))
        XYZ = XYZ[None,None,:] / 100
        sRGB = xyz2rgb(XYZ)
        return sRGB[0,0,:]


def get_image(shape, bg_colour):
    """
    :param shape: Desired image shape, e.g. (500,500,3)
    :param bg_colour: Desired background colour
    :return: Empty image with specified background colour
    """
    assert shape[2] == 3 # consistent with colour specifications
    image = np.zeros(shape, dtype=np.double)
    for i in range(3):
        image[:,:,i] = bg_colour[i]
    return image


def add_rectangle(image, centre, shape, angle, RGB):
    """
    Adds a rectangle to an image.

    :param image: An image
    :param centre: Pixel on which rectangle should be centred
    :param shape: (height, width) of rectangle
    :param angle: angle counterclockwise from right horizontal (radians)
    :param RGB: (red,green,blue) values between 0 and 1
    """

    corners = [[-shape[0]/2, -shape[1]/2],
               [shape[0]/2, -shape[1]/2],
               [shape[0]/2, shape[1]/2],
               [-shape[0]/2, shape[1]/2],
               [-shape[0] / 2, -shape[1] / 2]]
    rotation = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    rotated = np.dot(rotation, np.array(corners).transpose()).transpose()
    centered = np.add(rotated, centre).astype(np.int)
    rr, cc = polygon(centered[:,0], centered[:,1], image.shape)

    for i in range(3):
        image[rr, cc, i] = RGB[i]


def add_C(image, centre, shape, angle, RGB):
    """
    Adds a C-shape to an image, like those in Zhou et al.

    :param image: An image
    :param centre: Pixel on which C should be centred
    :param shape: (height, width) of C
    :param angle: angle counterclockwise from right horizontal (radians)
    :param RGB: (red,green,blue) values between 0 and 1
    """

    corners = [[-shape[0]/2, -shape[1]/2],
               [shape[0]/2, -shape[1]/2],
               [shape[0]/2, -shape[1]/4],
               [0, -shape[1]/4],
               [0, shape[1]/4],
               [shape[0]/2, shape[1]/4],
               [shape[0]/2, shape[1]/2],
               [-shape[0]/2, shape[1]/2],
               [-shape[0] / 2, -shape[1] / 2]]
    rotation = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    rotated = np.dot(rotation, np.array(corners).transpose()).transpose()
    centered = np.add(rotated, centre).astype(np.int)
    rr, cc = polygon(centered[:,0], centered[:,1], image.shape)

    for i in range(3):
        image[rr, cc, i] = RGB[i]


def get_preferred_stimulus(im_width, preferred_stimulus):
    """
    :param im_width: # of pixels
    :param preferred_stimulus: a dictionary with entries colour, length, width, angle (produced
        by experiment.find_optimal_bars)
    :return: stimulus image
    """
    preferred_colour = preferred_stimulus['colour']
    centre = im_width/2

    colours = Colours()
    bg_colour_name = 'Light gray (background)'
    bg_colour = colours.get_RGB(bg_colour_name, 0)

    stimulus_pref = get_image((im_width, im_width, 3), bg_colour)
    add_rectangle(stimulus_pref,
                  [centre,centre],
                  (preferred_stimulus['width'], preferred_stimulus['length']),
                  preferred_stimulus['angle'],
                  preferred_colour)
    return stimulus_pref


def get_standard_test_stimuli(im_width, preferred_stimulus):
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

    # Stimuli as in panels A-D of Zhou et al. Figure 2
    stimulus_A = get_image((im_width, im_width, 3), preferred_colour)
    add_rectangle(stimulus_A, position_1, square_shape, angle, bg_colour)

    stimulus_B = get_image((im_width, im_width, 3), bg_colour)
    add_rectangle(stimulus_B, position_2, square_shape, angle, preferred_colour)

    stimulus_C = get_image((im_width, im_width, 3), bg_colour)
    add_rectangle(stimulus_C, position_1, square_shape, angle, preferred_colour)

    stimulus_D = get_image((im_width, im_width, 3), preferred_colour)
    add_rectangle(stimulus_D, position_2, square_shape, angle, bg_colour)

    return stimulus_A, stimulus_B, stimulus_C, stimulus_D


def get_overlapping_squares_stimuli(im_width, preferred_stimulus):
    colours = Colours()
    bg_colour_name = 'Light gray (background)'
    bg_colour = colours.get_RGB(bg_colour_name, 0)

    preferred_colour = preferred_stimulus['colour']

    square_shape = (im_width/4, im_width/4)
    rectangle_shape = (1.3*im_width/4, im_width/4)

    angle = preferred_stimulus['angle']
    rotation = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    fg_offset_x = im_width/8
    fg_offset_y = -0.15*im_width/4
    bg_offset_x = fg_offset_x-0.15*im_width/4
    bg_offset_y = 0.15*im_width/4
    centre = im_width/2
    position_1 = np.add(np.dot(rotation, np.array([-fg_offset_x, fg_offset_y]).transpose()), [centre,centre]).astype(np.int)
    position_2 = np.add(np.dot(rotation, np.array([fg_offset_x, -fg_offset_y]).transpose()), [centre,centre]).astype(np.int)
    position_1b = np.add(np.dot(rotation, np.array([-bg_offset_x, -bg_offset_y]).transpose()), [centre,centre]).astype(np.int)
    position_2b = np.add(np.dot(rotation, np.array([bg_offset_x, bg_offset_y]).transpose()), [centre,centre]).astype(np.int)

    # Stimuli as in right 4 panels of Zhou et al. Figure 23
    stimulus_A = get_image((im_width, im_width, 3), colours.black)
    add_rectangle(stimulus_A, position_2b, rectangle_shape, angle, preferred_colour)
    add_rectangle(stimulus_A, position_1, square_shape, angle, bg_colour)

    stimulus_B = get_image((im_width, im_width, 3), colours.black)
    add_rectangle(stimulus_B, position_1b, rectangle_shape, angle, bg_colour)
    add_rectangle(stimulus_B, position_2, square_shape, angle, preferred_colour)

    stimulus_C = get_image((im_width, im_width, 3), colours.black)
    add_rectangle(stimulus_C, position_2b, rectangle_shape, angle, bg_colour)
    add_rectangle(stimulus_C, position_1, square_shape, angle, preferred_colour)

    stimulus_D = get_image((im_width, im_width, 3), colours.black)
    add_rectangle(stimulus_D, position_1b, rectangle_shape, angle, preferred_colour)
    add_rectangle(stimulus_D, position_2, square_shape, angle, bg_colour)

    return stimulus_A, stimulus_B, stimulus_C, stimulus_D


def get_c_shape_stimuli(im_width, preferred_stimulus):
    colours = Colours()
    bg_colour_name = 'Light gray (background)'
    bg_colour = colours.get_RGB(bg_colour_name, 0)

    print(preferred_stimulus)
    preferred_colour = preferred_stimulus['colour']

    c_shape = (im_width/4, im_width/2)

    angle = preferred_stimulus['angle']
    centre = im_width/2
    position = np.array([centre,centre]).astype(np.int)

    # Stimuli as in centre panels of Zhou et al. Figure 23
    stimulus_A = get_image((im_width, im_width, 3), bg_colour)
    add_C(stimulus_A, position, c_shape, angle+np.pi, preferred_colour)

    stimulus_B = get_image((im_width, im_width, 3), preferred_colour)
    add_C(stimulus_B, position, c_shape, angle, bg_colour)

    stimulus_C = get_image((im_width, im_width, 3), preferred_colour)
    add_C(stimulus_C, position, c_shape, angle+np.pi, bg_colour)

    stimulus_D = get_image((im_width, im_width, 3), bg_colour)
    add_C(stimulus_D, position, c_shape, angle, preferred_colour)

    return stimulus_A, stimulus_B, stimulus_C, stimulus_D


if __name__ == '__main__':
    colours = Colours()
    # bg_colour = colours.get_RGB('Light gray (background)', 0)
    # image = get_image((500, 500, 3), bg_colour)
    # add_C(image, (300,300), (100,200), .6*np.pi, colours.get_RGB('Red–brown', 0))
    # add_rectangle(image, (200,200), (100,200), .75*np.pi, colours.get_RGB('Blue-azure', 0))
    # plt.imshow(image)
    # plt.show()

    preferred_stimulus = {
        'colour': colours.get_RGB('Red–brown', 0),
        'length': 80,
        'width': 8,
        'angle': -np.pi * 1.25}
    # stimulus_A, stimulus_B, stimulus_C, stimulus_D = get_standard_test_stimuli(400, preferred_stimulus)
    stimulus_A, stimulus_B, stimulus_C, stimulus_D = get_overlapping_squares_stimuli(400, preferred_stimulus)
    # stimulus_A, stimulus_B, stimulus_C, stimulus_D = get_c_shape_stimuli(400, preferred_stimulus)

    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(stimulus_A)
    plt.subplot(2,2,2)
    plt.imshow(stimulus_C)
    plt.subplot(2,2,3)
    plt.imshow(stimulus_B)
    plt.subplot(2,2,4)
    plt.imshow(stimulus_D)

    for i in range(1,5):
        plt.subplot(2, 2, i)
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.show()

