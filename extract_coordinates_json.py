import json

import matplotlib.pyplot as plt


def _extract_json_coordinates_(shapes, dip_to_mm_factor=0.15875):
    """
    Usage
    -----
    Helper function to scaled extract coordinates from the json object.

    Parameters
    ----------
    param x_coord : x_coord of respective shapes
    param y_coord : y_coord of respective shapes
    idx : json object of the json file
    param dip_to_mm_factor : (scalar) the conversion factor from density independent pixels to milimeter.

    Returns
    -------
    x_coord : a list of list of x coordinates according to the breaks taken by patient
    y_coord : a list of list of y coordinates according to the breaks taken by patient
    """
    temp_list = []
    for breaks in shapes['path']:
        temp_list.append(breaks['path']['data'])

    x_coord = [[dip_to_mm_factor * float(j.split(',')[0]) for j in i] for i in temp_list]
    y_coord = [[dip_to_mm_factor * float(j.split(',')[1]) for j in i] for i in temp_list]

    return x_coord, y_coord


def extract_json_coordinates(data, number_of_try):
    """
    Usage
    -----
    Extract coordinates and related details like the mobile canvas (dimension), given the i-th try by user.

    Parameters
    ----------
    param data : (dict) dictionary of json file.
    param number_try : the i-th try by user.
    param dip_to_mm_factor : (scalar) the conversion factor from density independent pixels to millimeter.

    Returns
    -------
    output_dictionary_user_details : a dictionary with relevant details for the patient
    """
    output_dictionary_user_details = {}

    for shapes in data[number_of_try]['related_details']:
        if shapes['name'] == 'shape1':
            x_coord, y_coord = _extract_json_coordinates_(shapes)
            output_dictionary_user_details['shape1_coord_user'] = {'x_coord': x_coord,
                                                                   'y_coord': y_coord}
        elif shapes['name'] == 'shape2':
            x_coord, y_coord = _extract_json_coordinates_(shapes)
            output_dictionary_user_details['shape2_coord_user'] = {'x_coord': x_coord,
                                                                   'y_coord': y_coord}
        else:
            x_coord, y_coord = _extract_json_coordinates_(shapes)
            output_dictionary_user_details['shape3_coord_user'] = {'x_coord': x_coord,
                                                                   'y_coord': y_coord}
    return output_dictionary_user_details


# if __name__ == '__main__':
#
#     file_path = r'/home/yash/Desktop/HaTran/zakuta.json'
#     with open(file_path) as data_file:
#         data = json.load(data_file)
#
#     out = extract_json_coordinates_and_related_details(data, 2)
#     x = out['shape3_coord_user']['x_coord']
#     y = out['shape3_coord_user']['y_coord']
#
#     for i in range(len(x)):
#         plt.plot(x[i], y[i], color='black', linewidth=2)
#
#     plt.axis('off')
#     # plt.savefig('patient_input_0.png', format='png')
#     plt.show()


def save_patient_image(json_path, shape_type, number_of_try):

    with open(json_path) as file:
        data = json.load(file)

    out = extract_json_coordinates(data, number_of_try)

    x = out[f'{shape_type}_coord_user']['x_coord']
    y = out[f'{shape_type}_coord_user']['y_coord']

    if len(x) != len(y):
        raise ValueError('Error while extracting the coordinates from the json file.')

    for i in range(len(x)):
        plt.plot(x[i], y[i], color='black', linewidth=2, marker='.', markersize=1)

    plt.axis('off')
    # plt.savefig(f'data/patient_inp_{shape_type}_{number_of_try}.png', format='png')
    plt.show()
    # plt.close()


def get_patient_status(json_data, number_of_try):

    status = json_data[number_of_try]['status']

    return status


if __name__ == '__main__':

    save_patient_image('/home/yash/Desktop/HaTran/test_case_confluence/data/1st_assessment_with_training.json',
                       shape_type='shape3', number_of_try=0)
