import torch
import json
import numpy as np
from scipy import misc
from scipy.ndimage.filters import gaussian_filter
from matplotlib import pyplot as plt
from time import time
from statsmodels import robust
from scipy.spatial import distance

import extract_coordinates_json as json_coord
import extract_svg_coordinates as svg_coord

use_cuda = torch.cuda.is_available()
tensor   = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

plt.ion()
plt.show()

s2v = lambda x: tensor([x])

svg_path_shape1 = r'/data/Shape1_start_absolute.svg'
svg_path_shape2 = r'/data/Shape2_start_absolute.svg'
svg_path_shape3 = r'/data/Shape3_start_absolute.svg'

json_path = r'/data/1st_assessment_with_training.json'


# Loop on all the experiments ================================================================
experiments = {}

if True:  # Sinkhorn
    for p in [2]:  # C(x,y) = |x-y|^1 or |x-y|^2
        for eps, eps_s in [(.01, "S")]:
            for nits in [2]:
                experiments["sinkhorn_L{}_{}_{}its".format(p, eps_s, nits)] = {
                    "formula": "sinkhorn",
                    "p": p,
                    "eps": eps ** p,  # Remember : eps is homogeneous to C(x,y)
                    "nits": nits,
                    "tol": 0.,  # Run all iterations, no early stopping!
                    "transport_plan": "heatmaps",
                }


# Load the pngs ================================================================================

def LoadImage(fname):

    img = misc.imread(fname, flatten=True)    # Grayscale
    img = gaussian_filter(img, 1, mode='nearest')
    img = (img[::-1, :]) / 255.
    img = np.swapaxes(img, 0, 1)
    return tensor(1 - img)


def extract_point_cloud(I, affine) :
    """Bitmap to point cloud."""

    # Threshold, to extract the relevant indices ---------------------------------------
    ind = (I > .001).nonzero()

    # Extract the weights --------------------------------------------------------------
    D = len(I.shape)
    if   D == 2 : α_i = I[ind[:,0], ind[:,1]]
    elif D == 3 : α_i = I[ind[:,0], ind[:,1], ind[:,2]]
    else : raise NotImplementedError()

    α_i = α_i * affine[0,0] * affine[1,1] # Lazy approximation of the determinant...
    # If we normalize the measures, it doesn't matter anyway.

    # Don't forget the changes of coordinates! -----------------------------------------
    M   = affine[:D,:D] ; off = affine[:D,D]
    x_i = ind.float() @ M.t() + off

    return ind, α_i.view(-1, 1), x_i


def _calculate_score_(json_path, number_of_try, shape_type, template_path, name, params, verbose=False):

    t_0 = time()
    svg_coord.save_template_image(svg_path=template_path, shape_type=shape_type)
    plt.pause(0.5)
    json_coord.save_patient_image(json_path=json_path, shape_type=shape_type, number_of_try=number_of_try)

    dataset = "shape"
    datasets = {
        "shape": (f"data/patient_inp_{shape_type}_{number_of_try}.png", f"data/{shape_type}.png"),
    }

    # Note that both measures will be normalized in "sparse_distance_bmp"
    source = LoadImage(datasets[dataset][0])
    target = LoadImage(datasets[dataset][1])

    # The images are rescaled to fit into the unit square ==========================================
    scale = source.shape[0]
    affine = tensor([[1, 0, 0], [0, 1, 0]]) / scale

    ind_source, α_i_source, x_i_source = extract_point_cloud(source, affine)
    ind_target, α_i_target, x_i_target = extract_point_cloud(target, affine)

    distance_matrix = distance.cdist(x_i_source, x_i_target, 'euclidean')

    sq_min = []
    for i in range(len(distance_matrix)):
        sq_min.append(np.min(distance_matrix[i]))

    answer = np.mean(sq_min)

    # We'll save the output wrt. the number of iterations
    # display = False
    #
    # cost, grad_src, heatmaps = sparse_distance_bmp(params, source, target, affine, affine, normalize=True,
    #                                                info=display)
    # t_1 = time()
    # if verbose:
    #     print("{} : {:.2f}s, cost = {:.6f}".format(name, t_1-t_0, cost.item()))

    return float("{:.6f}".format(answer))


def calculate_score(shape_type, number_of_try, template_path):

    for name, params in experiments.items():
        cost = _calculate_score_(json_path=json_path, number_of_try=number_of_try, shape_type=shape_type,
                                 template_path=template_path, name=name, params=params)

    return cost


def get_training_score(json_data, shape_type, template_path):

    train_score_array = []

    for tries in range(len(json_data)):
        status = json_coord.get_patient_status(json_data=json_data, number_of_try=tries)
        if status == 'training':
            cost = calculate_score(number_of_try=tries, shape_type=shape_type, template_path=template_path)
            train_score_array.append(cost)

    return train_score_array


def get_tracking_score(json_data, shape_type, template_path):

    track_score = []

    for tries in range(len(json_data)):
        status = json_coord.get_patient_status(json_data=json_data, number_of_try=tries)
        if status == 'tracking':
            cost = calculate_score(number_of_try=tries, shape_type=shape_type, template_path=template_path)
            track_score.append(cost)

    return track_score


def classify_track_score(train_score_array, track_score):

    median = np.median(train_score_array)
    mad = robust.mad(train_score_array)

    if track_score < median:
        score = 0
    elif median <= track_score <= median + mad:
        score = 0
    elif median + mad < track_score <= median + (2 * mad):
        score = 1
    elif median + (2 * mad) < track_score <= median + (3 * mad):
        score = 2
    else:
        score = 3

    return score, median, mad


def push_notification_normal(train_median, train_mad, normal, track_score, shape_type):

    threshold = train_median + train_mad
    normalized_normal = normal * 100
    threshold_normal = threshold + threshold * normalized_normal
    if threshold < track_score:
        if threshold_normal < track_score:
            print('notify the physician for ' + f'{shape_type}.')


def run(json_data):

    t_0 = time()
    training_score_array_shape1 = get_training_score(json_data=json_data, shape_type='shape1',
                                                     template_path=svg_path_shape1)
    training_score_array_shape2 = get_training_score(json_data=json_data, shape_type='shape2',
                                                     template_path=svg_path_shape2)
    training_score_array_shape3 = get_training_score(json_data=json_data, shape_type='shape3',
                                                     template_path=svg_path_shape3)

    tracking_score_shape1 = get_tracking_score(json_data=json_data, shape_type='shape1',
                                               template_path=svg_path_shape1)
    tracking_score_shape2 = get_tracking_score(json_data=json_data, shape_type='shape2',
                                               template_path=svg_path_shape2)
    tracking_score_shape3 = get_tracking_score(json_data=json_data, shape_type='shape3',
                                               template_path=svg_path_shape3)

    shape1_score, median_shape1, mad_shape1 = classify_track_score(train_score_array=training_score_array_shape1,
                                                                   track_score=tracking_score_shape1)
    shape2_score, median_shape2, mad_shape2 = classify_track_score(train_score_array=training_score_array_shape2,
                                                                   track_score=tracking_score_shape2)
    shape3_score, median_shape3, mad_shape3 = classify_track_score(train_score_array=training_score_array_shape3,
                                                                   track_score=tracking_score_shape3)

    push_notification_normal(train_median=median_shape1, train_mad=mad_shape1, track_score=tracking_score_shape1,
                             shape_type='shape1', normal=0.33)
    push_notification_normal(train_median=median_shape2, train_mad=mad_shape2, track_score=tracking_score_shape2,
                             shape_type='shape2', normal=0.33)
    push_notification_normal(train_median=median_shape3, train_mad=mad_shape3, track_score=tracking_score_shape3,
                             shape_type='shape3', normal=0.33)

    t_1 = time()
    total_time = t_1 - t_0

    return shape1_score, shape2_score, shape3_score, total_time


if __name__ == '__main__':

    with open(json_path) as file:
        data = json.load(file)

    score1, score2, score3, total_time = run(json_data=data)
    print(score1, score2, score3, total_time)
