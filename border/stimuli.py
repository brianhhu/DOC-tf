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


if __name__ == '__main__':
    colours = Colours()
    bg_colour = colours.get_RGB('Light gray (background)', 0)

    image = get_image((500, 500, 3), bg_colour)
    add_C(image, (300,300), (100,200), .6*np.pi, colours.get_RGB('Red–brown', 0))
    add_rectangle(image, (200,200), (100,200), .75*np.pi, colours.get_RGB('Blue-azure', 0))
    plt.imshow(image)
    plt.show()