import numpy as np
from svgpathtools import svg2paths, Path, CubicBezier, Line
import matplotlib.pyplot as plt


def _extract_svg_coordinates_helper_function_(paths, number_of_samples=30):
    """
    Usage
    -----
    Helper function to extract coordinates from the paths given in the drawing svg files using svgpathtools.
    Parameters
    ----------
    param paths : a list of paths in the svg file.
    param number_of_samples : (scalar) the sampling rate at which the points should be drawn to avoid discontinuity in
    coordinates of the drawing.

    Returns
    -------
    numpy.array : a 2D array of x and y coordinates.
    """
    path_coordinates = []
    x_coord = []
    y_coord = []

    for idx in paths:
        for jdy in idx:
            for j in range(number_of_samples):
                path_coordinates.append(jdy.point(j / (number_of_samples - 1)))

    for k in range(len(path_coordinates)):
        xi = path_coordinates[k].real
        yi = path_coordinates[k].imag

        x_coord.append(xi)
        y_coord.append(yi)

    return list(zip(np.asarray(x_coord), np.asarray(y_coord)))


def extract_svg_coordinates(path_to_file, shape_type):
    """
    Usage
    -----
    Extracts the coordinates of the svg file according to the shape_type
    ----------
    param shape_type : (str) three shape types = ['shape1', 'shape2', 'shape3']
    param path_to_file : (str) path to the svg shape file for which the coordinates are to be extracted.

    Returns
    -------
    dict : a dictionary of coordinates of the shape.
    """
    # paths, attributes = svg2paths('/home/yash/Desktop/HaTran/Shape1.svg')
    paths, attributes = svg2paths(path_to_file)

    shape1_coord_template = []
    shape2_coord_template = []
    shape3_coord_template = []
    output_dictionary_template_details = {}

    if shape_type == 'shape1':
        shape1_coord_template = _extract_svg_coordinates_helper_function_(paths)
    elif shape_type == 'shape2':
        shape2_coord_template = _extract_svg_coordinates_helper_function_(paths)
    elif shape_type == 'shape3':
        shape3_coord_template = _extract_svg_coordinates_helper_function_(paths)

    output_dictionary_template_details['shape1_coord_template'] = shape1_coord_template
    output_dictionary_template_details['shape2_coord_template'] = shape2_coord_template
    output_dictionary_template_details['shape3_coord_template'] = shape3_coord_template

    return output_dictionary_template_details


def save_template_image(svg_path, shape_type):

    out = extract_svg_coordinates(svg_path, shape_type)

    x = []
    y = []

    for i in range(len(out[f'{shape_type}_coord_template'])):
        x.append(out[f'{shape_type}_coord_template'][i][0])
        y.append(out[f'{shape_type}_coord_template'][i][1])

    plt.plot(x, y, color='black', linewidth=2)
    plt.axis('off')
    plt.savefig(f'template_{shape_type}.png', format='png')
    plt.close()
#
# if __name__ == '__main__':
#
#     svg_path = r'/home/yash/Desktop/HaTran/Shape1_start_absolute.svg'
#     save_template_image(svg_path=svg_path, shape_type='shape1')
