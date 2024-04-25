import threading

import numpy as np
import cv2
import csv
import os
import math
import scipy.ndimage
from numpy import linalg as LA
from scipy.spatial import distance_matrix
import argparse
from moviepy.editor import *
import matplotlib.pyplot as plt
import pickle
import collections
import pandas as pd
from threading import Thread
import time
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
from concurrent.futures import ThreadPoolExecutor
plt.style.use('classic')
from datetime import datetime


def find_closest_match(locust_merge_dict, dist_between_same_trajectories=150, frames_between_same_trajectories=1500):
    """
    Find and return a dictionary of closest matching trajectories.

    :param locust_merge_dict: Dictionary containing locust trajectories.
    :param dist_between_same_trajectories: Distance threshold for merging trajectories.
    :param frames_between_same_trajectories: Frame threshold for merging trajectories.
    :return: Dictionary with closest matching trajectories.
    """
    min_dist = np.inf
    locust_temp_merge_dict = {}
    for key in list(locust_merge_dict.keys()):
        for other_key in list(locust_merge_dict.keys()):
            if other_key != key:
                dist_end_to_start = np.sqrt(
                    np.power(locust_merge_dict[key][4] - locust_merge_dict[other_key][1], 2) + np.power(
                        locust_merge_dict[key][5] - locust_merge_dict[other_key][2], 2))
                dist_frames_end_to_start = locust_merge_dict[other_key][0] - locust_merge_dict[key][3]
                if dist_frames_end_to_start > 0 and dist_frames_end_to_start < frames_between_same_trajectories:
                    if (dist_end_to_start < dist_between_same_trajectories) and dist_end_to_start < min_dist:
                        min_dist = dist_end_to_start
                        min_key_to_merge = other_key
        if min_dist < dist_between_same_trajectories:
            if min_key_to_merge not in list(locust_temp_merge_dict.values()):
                locust_temp_merge_dict[key] = min_key_to_merge
        min_dist = np.inf
    return locust_temp_merge_dict


def merge_locusts(locust_temp_merge_dict, locust_traj_list, locust_merge_dict):
    """
    Merge locust trajectories based on the provided merge dictionary.

    :param locust_temp_merge_dict: Dictionary containing locusts to be merged.
    :param locust_traj_list: List of locust trajectories.
    :param locust_merge_dict: Dictionary containing locust merge information.
    :return: Tuple containing updated locust_merge_dict and locust_traj_list.
    """
    for key in list(locust_temp_merge_dict.keys()):

        df_1 = [f for f in locust_traj_list if f.head(1)['Locust_ID'].values[0] == key][0]
        ind_1 = [index for index, value in enumerate(locust_traj_list) if value is df_1][0]
        df_2 = [f for f in locust_traj_list if f.head(1)['Locust_ID'].values[0] == locust_temp_merge_dict[key]][0]
        ind_2 = [index for index, value in enumerate(locust_traj_list) if value is df_2][0]
        df_1 = df_1.append(df_2)
        locust_traj_list[ind_1] = df_1
        del locust_traj_list[ind_2]

        # Now we need to update the locust_merge_dict for the locust that we merged the data into and delete the unwanted one
        locust_merge_dict[df_1['Locust_ID'].head(1).values[0]] = [int(df_1.head(1)['Frame_Number'].values[0]),
                                                                  int(df_1.head(1)['Location_X'].values[0]),
                                                                  int(df_1.head(1)['Location_Y'].values[0]),
                                                                  int(df_1.tail(1)['Frame_Number'].values[0]),
                                                                  int(df_1.tail(1)['Location_X'].values[0]),
                                                                  int(df_1.tail(1)['Location_Y'].values[0])]
        del locust_merge_dict[df_2['Locust_ID'].head(1).values[0]]
    return locust_merge_dict, locust_traj_list

def merge_trajectories(dir_name):
    files_in_folder = os.listdir(dir_name)
    files_in_folder = [f for f in files_in_folder if (f.startswith('locust') and f.endswith('.csv'))]
    locust_traj_list = []

    locust_merge_dict = {}

    for i in range(len(files_in_folder)):
        locust_traj_file = pd.read_csv(dir_name + '/' + files_in_folder[i])
        if len(locust_traj_file) > 0:
            locust_traj_list.append(locust_traj_file)
            first_frame_in_traj = int(locust_traj_file['Frame_Number'].T[0])
            location_x_first_frame_in_traj = locust_traj_file['Location_X'].T[0]
            location_y_first_frame_in_traj = locust_traj_file['Location_Y'].T[0]
            last_frame_in_traj = int(locust_traj_file['Frame_Number'].T[len(locust_traj_file['Frame_Number']) - 1])
            location_x_last_frame_in_traj = locust_traj_file['Location_X'].T[len(locust_traj_file['Frame_Number']) - 1]
            location_y_last_frame_in_traj = locust_traj_file['Location_Y'].T[len(locust_traj_file['Frame_Number']) - 1]
            locust_merge_dict[locust_traj_file['Locust_ID'].T[0]] = [first_frame_in_traj,
                                                                     location_x_first_frame_in_traj,
                                                                     location_y_first_frame_in_traj, last_frame_in_traj,
                                                                     location_x_last_frame_in_traj,
                                                                     location_y_last_frame_in_traj]

    locust_merge_dict = collections.OrderedDict(sorted(locust_merge_dict.items()))
    locust_temp_merge_dict = find_closest_match(locust_merge_dict, dist_between_same_trajectories=150,
                                                frames_between_same_trajectories=1500)
    locust_temp_merge_dict = collections.OrderedDict(reversed(sorted(locust_temp_merge_dict.items())))
    locust_merge_dict, locust_traj_list = merge_locusts(locust_temp_merge_dict, locust_traj_list, locust_merge_dict)
    locust_temp_merge_dict = find_closest_match(locust_merge_dict, dist_between_same_trajectories=200,
                                                frames_between_same_trajectories=2500)
    locust_temp_merge_dict = collections.OrderedDict(reversed(sorted(locust_temp_merge_dict.items())))
    locust_merge_dict, locust_traj_list = merge_locusts(locust_temp_merge_dict, locust_traj_list, locust_merge_dict)
    locust_temp_merge_dict = find_closest_match(locust_merge_dict, dist_between_same_trajectories=250,
                                                frames_between_same_trajectories=3500)
    locust_temp_merge_dict = collections.OrderedDict(reversed(sorted(locust_temp_merge_dict.items())))
    locust_merge_dict, locust_traj_list = merge_locusts(locust_temp_merge_dict, locust_traj_list, locust_merge_dict)

    for i in range(len(locust_traj_list)):
        locust_df= locust_traj_list[i]
        locust_df.to_excel(dir_name +  '\locust_file_' + str(locust_df['Locust_ID'].head(1).values[0]).zfill(2) + '.xlsx')
    return locust_traj_list


def click_event(event, x, y, flags, param):
    """
    Capture mouse click events to define coordinates on an image.

    :param event: Mouse event (e.g., cv2.EVENT_LBUTTONDOWN).
    :param x: X-coordinate of the mouse click.
    :param y: Y-coordinate of the mouse click.
    :param flags: Additional flags.
    :param param: Additional parameters.
    :return: None
    """

    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,",",y)
        maze_coordinates.append([x,y])
        font = cv2.FONT_HERSHEY_SIMPLEX
        strXY = str(x)+", "+str(y)
        cv2.putText(img, strXY, (x,y), font, 0.5, (255,255,0), 2)
        cv2.imshow("LocusTracker", img)

        if len(maze_coordinates) == 1:
            print('Click on upper right corner of maze')
        elif len(maze_coordinates) == 2:
            print('Click on lower left corner of maze')
        elif len(maze_coordinates) == 3:
            print('Click on lower right corner of maze')
        else:
            print('All coordinates selected successfully!')
            print('Proceeding to trajectory calculation phase!')
            cv2.destroyAllWindows()


def get_cropped_maze_region(cropped_image, maze_coordinates):
    """
    Crop the input image to the region defined by the provided maze coordinates.

    :param cropped_image: The original image to be cropped.
    :param maze_coordinates: List of coordinates defining the cropping region.
    :return: Cropped image and the bounds of the cropped region (min_y, max_y, min_x, max_x).
    """
    maze_coordinates = np.asarray(maze_coordinates)
    maze_coordinates_x = maze_coordinates[:, 0]
    maze_coordinates_y = maze_coordinates[:, 1]

    min_y, max_y = np.min(maze_coordinates_y), np.max(maze_coordinates_y)
    min_x, max_x = np.min(maze_coordinates_x), np.max(maze_coordinates_x)

    cropped_image = cropped_image[min_y:max_y, min_x:max_x, :]

    return cropped_image, min_y, max_y, min_x, max_x

def calc_distances(p0, points):
    """
    Calculate the L2 distance between a point p0 and a set of points.

    :param p0: The reference point.
    :param points: A set of points to calculate distances from.
    :return: Array of L2 distances.
    """
    return np.sum((p0 - points) ** 2, axis=1)


def furthest_point_sampling(pts, k):
    """
    Perform furthest point sampling to select k points from a group of points.

    :param pts: Array of input points.
    :param k: Number of points to sample.
    :return: Array of k sampled points.
    """
    furthest_pts = np.zeros((k, 2))
    furthest_pts[0] = pts[np.random.randint(len(pts))]

    distances = calc_distances(furthest_pts[0], pts)

    for i in range(1, k):
        furthest_pts[i] = pts[np.argmax(distances)]
        distances = np.minimum(distances, calc_distances(furthest_pts[i], pts))

    return furthest_pts


def compute_blobs(diffim, radius, city_name, white_area_list):
    """
    Calculate the center of mass for each locust and return its coordinates.

    :param diffim: Difference image between the current frame and the background image.
    :param radius: Radius corresponding to locust size.
    :param city_name: Name of the city used to select the threshold based on the city density.
    :param white_area_list: List of white areas that should be ignored.
    :return: Tuple containing blob_centers_list, diffim, blurred_im, is_cluster_frame, number_of_locusts_frame.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX  # Font type
    fontScale = 0.75  # Font scale
    thickness = 2  # Thickness in pixels
    locust_area = 120
    kernel = np.zeros((2 * radius + 1, 2 * radius + 1))
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    mask = x ** 2 + y ** 2 <= radius ** 2

    kernel[mask] = 1
    kernel[mask] = 1 / ((math.pi) * math.pow(radius, 2))
    diffim = diffim / np.max(diffim)
    diffim = diffim * 255

    ny_city_thresh = 75
    cairo_city_thresh = 100
    rome_city_thresh = 100

    if city_name == 'new_york':
        city_thresh = ny_city_thresh
    elif city_name == 'cairo':
        city_thresh = cairo_city_thresh
    else:
        city_thresh = rome_city_thresh

    ret, thresh_new = cv2.threshold(diffim, city_thresh, 255, cv2.THRESH_TOZERO)

    blurred_im = cv2.GaussianBlur(thresh_new, (radius, radius), 0).astype(np.uint8)

    ret, blurred_im = cv2.threshold(blurred_im, 45, 255, cv2.THRESH_TOZERO)

    is_cluster_frame = np.array([])
    number_of_locusts_frame = np.array([])

    if ret:
        im_contours, contours = cv2.findContours(blurred_im.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(im_contours) > 0:
            blob_centers_list = np.array([])

            for c in im_contours:
                M = cv2.moments(c)
                area = cv2.contourArea(c)
                mask = np.zeros(blurred_im.shape, np.uint8)
                cv2.drawContours(mask, [c], 0, 255, -1)

                if M["m00"] > 0 and area > 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])

                    if len(np.argwhere((white_area_list[:, 0] == cX) & (white_area_list[:, 1] == cY))) == 0:
                        blob_centers_list = np.append(blob_centers_list, [cX, cY])
                        cv2.circle(diffim, (cX, cY), 3, (255, 255, 255), -1)

                if area > locust_area and M["m00"] > 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])

                    if len(np.argwhere((white_area_list[:, 0] == cX) & (white_area_list[:, 1] == cY))) == 0:
                        number_of_locust_in_cluster = int(np.maximum(np.floor(area / locust_area), 1))

                        if number_of_locust_in_cluster > 1:
                            is_cluster_frame = np.append(is_cluster_frame, 1)
                        else:
                            is_cluster_frame = np.append(is_cluster_frame, 0)

                        number_of_locusts_frame = np.append(number_of_locusts_frame, number_of_locust_in_cluster)

                elif M["m00"] > 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])

                    if len(np.argwhere((white_area_list[:, 0] == cX) & (white_area_list[:, 1] == cY))) == 0:
                        is_cluster_frame = np.append(is_cluster_frame, 0)
                        number_of_locusts_frame = np.append(number_of_locusts_frame, 1)

            if len(blob_centers_list) > 0:
                blob_centers_list = blob_centers_list.reshape(-1, 1, 2)
                return (
                    blob_centers_list.astype(np.float32),
                    diffim.astype(np.uint8),
                    blurred_im.astype(np.uint8),
                    is_cluster_frame.astype(int),
                    number_of_locusts_frame.astype(int)
                )
            else:
                return None, None, None, None, None
        else:
            return None, None, None, None, None
    else:
        return None, None, None, None, None

def difference_matrix(a):
    """
    Generate a matrix representing the pairwise differences between elements in the input array.

    :param a: Input array.
    :return: Difference matrix.
    """
    x = np.reshape(a, (len(a), 1))
    return x - x.transpose()


def calculate_active_area(city_template_im, background_im, dir_name):
    """
    Calculate the active area (area that locust can move in) in percentage and save this information in a CSV file.

    :param city_template_im: City template image.
    :param background_im: Background image.
    :param dir_name: Directory name for saving the CSV file.
    :return: Pixels to centimeters conversion factor.
    """
    pixels_2_cm_background_im = 120 / (np.mean(background_im.shape) * (1 / scale))

    template_im_resized = cv2.resize(city_template_im, (background_im.shape[1], background_im.shape[0]))

    wall_pixels = np.sum(np.sum(template_im_resized))
    active_area_pixels = (background_im.shape[1] * background_im.shape[0]) - wall_pixels
    active_area_cm = active_area_pixels * np.power(pixels_2_cm_background_im, 2)
    total_area_cm = (background_im.shape[1] * background_im.shape[0]) * np.power(pixels_2_cm_background_im, 2)
    active_area_percent = (active_area_cm / total_area_cm) * 100
    active_area_percent_text = f"active_area_percent, {active_area_percent:.2f} %"
    print(active_area_percent_text)

    # Save the active area percentage to a CSV file
    with open(dir_name + '/active_area.csv', mode='w', newline='') as area_file:
        area_writer = csv.writer(area_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        area_writer.writerow(['active_area_percent'])
        area_writer.writerow([np.around(active_area_percent, decimals=2)])

    return pixels_2_cm_background_im

def save_heatmaps_info(location_time_counter, accumulated_image, location_center_time_counter, dir_name, frame_counter):
    """
    Save information related to location time counters and heatmaps.

    :param location_time_counter: Counter for locust locations over time.
    :param accumulated_image: Accumulated heatmap image.
    :param location_center_time_counter: Counter for locust center locations over time.
    :param dir_name: Directory name for saving files.
    :param frame_counter: Counter for frames.
    """
    # Normalize the accumulated image for display
    accumulated_image_opencv = (accumulated_image / frame_counter) * 255

    # Plot and save the heatmap using matplotlib
    fig = plt.figure()
    plt.imshow(accumulated_image)
    plt.colorbar(label='Number of Frames')
    plt.title('Locust Location Heatmap')
    plt.savefig(dir_name + '\matplotlib_heatmap.png')

    # Save the heatmap image using OpenCV
    cv2.imwrite(dir_name + '\heatmap.png', accumulated_image_opencv)

    # Save location time counter to a pickle file
    location_time_counter_pickle_path = dir_name + '\location_time_counter.pickle'
    with open(location_time_counter_pickle_path, 'wb') as pickle_output:
        pickle.dump(location_time_counter, pickle_output)

    # Save location center time counter to a pickle file
    location_center_time_counter_pickle_path = dir_name + '\location_center_time_counter.pickle'
    with open(location_center_time_counter_pickle_path, 'wb') as pickle_output_center:
        pickle.dump(location_center_time_counter, pickle_output_center)

def compute_heading(current_movement_points, trajlist, closest_inds, i):
    """
    Compute the heading of a locust compared with its current and previous location.

    :param current_movement_points: Array containing current locust movement points.
    :param trajlist: List of locust trajectories.
    :param closest_inds: Array containing indices of closest trajectories.
    :param i: Index of the locust.
    :return: Tuple containing current heading and previous heading.
    """
    # Compute the heading with respect to the current and previous locations
    current_heading = np.degrees(np.arctan(
        (current_movement_points[i, 0][0] - trajlist[closest_inds[i]][-1][0]) /
        (current_movement_points[i, 0][1] - trajlist[closest_inds[i]][-1][1])
    ))
    prev_heading = np.degrees(np.arctan(
        (trajlist[closest_inds[i]][-1][0] - trajlist[closest_inds[i]][-2][0]) /
        (trajlist[closest_inds[i]][-1][1] - trajlist[closest_inds[i]][-2][1])
    ))

    return current_heading, prev_heading

def add_points_to_trajectory(trajlist, framelist, is_cluster, number_of_locusts, max_movement, max_stationary,
                             closest_inds, i, frame_counter, current_movement_points, closest_dists,
                             previous_movement_points, max_locusts, is_cluster_frame, number_of_locusts_frame,
                             unassigned_points):
    """
    Add points to the formed trajectories of each locust.

    :param trajlist: List of locust trajectories.
    :param framelist: List of corresponding frames for each locust trajectory.
    :param is_cluster: List indicating whether a locust is part of a cluster.
    :param number_of_locusts: List indicating the number of locusts in each trajectory.
    :param max_movement: Maximum allowed movement for a locust to be considered part of the same trajectory.
    :param max_stationary: Maximum allowed stationary frames for a locust to be considered part of the same trajectory.
    :param closest_inds: Array containing indices of closest trajectories.
    :param i: Index of the locust.
    :param frame_counter: Current frame number.
    :param current_movement_points: Array containing current locust movement points.
    :param closest_dists: Array containing distances to the closest trajectories.
    :param previous_movement_points: Array containing previous locust movement points.
    :param max_locusts: Maximum number of locusts allowed.
    :param is_cluster_frame: Array indicating whether a locust is part of a cluster for the current frame.
    :param number_of_locusts_frame: Array indicating the number of locusts for each trajectory in the current frame.
    :param unassigned_points: Array containing unassigned movement points.
    :return: Updated trajlist, framelist, is_cluster, number_of_locusts, and unassigned_points.
    """

    if (closest_dists[i] < max_movement) and np.array(framelist[closest_inds[i]]).size < max_stationary:
        # Add point to existing trajectory if within maximum movement and stationary limits
        trajlist[closest_inds[i]].append(current_movement_points[i, 0, :])
        framelist[closest_inds[i]].append(frame_counter)
        is_cluster[closest_inds[i]].append(is_cluster_frame[i])
        number_of_locusts[closest_inds[i]].append(number_of_locusts_frame[i])
        # Remove the assigned point from unassigned points
        idx = unassigned_points != current_movement_points[i, 0, :]
        un_ind = [(idx[k][0][0] | idx[k][0][1]) for k in range(len(idx))]
        unassigned_points =  unassigned_points[un_ind]
    elif (closest_dists[i] < max_movement) and (frame_counter - framelist[closest_inds[i]][-1] < max_stationary):
        # Add point to existing trajectory if within maximum movement and stationary limits, but frame limit is not reached
        trajlist[closest_inds[i]].append(current_movement_points[i, 0, :])
        framelist[closest_inds[i]].append(frame_counter)
        is_cluster[closest_inds[i]].append(is_cluster_frame[i])
        number_of_locusts[closest_inds[i]].append(number_of_locusts_frame[i])
        # Remove the assigned point from unassigned points
        idx = unassigned_points != current_movement_points[i, 0, :]
        un_ind = [(idx[k][0][0] | idx[k][0][1]) for k in range(len(idx))]
        unassigned_points = unassigned_points[un_ind]
    elif len(trajlist) < max_locusts:
        # Old locusts are lost, and a new one is detected far away from the old ones, so assign it a new track
        trajlist.append([current_movement_points[i, 0, :], previous_movement_points[i, 0, :]])
        framelist.append([frame_counter])
        is_cluster.append([is_cluster_frame[i]])
        number_of_locusts.append([number_of_locusts_frame[i]])
        # Remove the assigned point from unassigned points
        idx = unassigned_points != current_movement_points[i, 0, :]
        un_ind = [(idx[k][0][0] | idx[k][0][1]) for k in range(len(idx))]
        unassigned_points = unassigned_points[un_ind]
    elif closest_dists[i] < (2 * max_movement):
        # Add point to existing trajectory if within twice the maximum movement limit
        trajlist[closest_inds[i]].append(current_movement_points[i, 0, :])
        framelist[closest_inds[i]].append(frame_counter)
        is_cluster[closest_inds[i]].append(is_cluster_frame[i])
        number_of_locusts[closest_inds[i]].append(number_of_locusts_frame[i])
        # Remove the assigned point from unassigned points
        idx = unassigned_points != current_movement_points[i, 0, :]
        un_ind = [(idx[k][0][0] | idx[k][0][1]) for k in range(len(idx))]
        unassigned_points = unassigned_points[un_ind]

    return trajlist, framelist, is_cluster, number_of_locusts, unassigned_points

def find_closest_n_locusts_in_prev_frame(reference_locust_location, n, prev_frame_points):
    """
    Find the n closest locusts to a reference locust's location.

    :param reference_locust_location: Location of the reference locust.
    :param n: Number of closest locusts to find.
    :param prev_frame_points: Points representing locust locations in the previous frame.
    :return: List of indices of the n closest locusts.
    """

    closest_locusts = []
    max_allowable_merge_split_dist = 50

    if len(prev_frame_points) == 0:
        print('No points in the previous frame.')

    # Calculate distances and get the indices sorted in ascending order
    dist_order = LA.norm(prev_frame_points - reference_locust_location, axis=1)
    ind_order = np.argsort(dist_order)

    # Find the n closest locusts
    for j in range(n):
        if j + 1 < len(ind_order):
            n_closest_locust = ind_order[j + 1]
            if dist_order[j + 1] < max_allowable_merge_split_dist:
                closest_locusts.append(n_closest_locust)
        else:
            closest_locusts.append(-1)

    return closest_locusts

def find_closest_n_locusts_in_frame(reference_locust_location, n, movement_points, max_locusts):
    """
    Find the n closest locusts to a reference locust's location.

    :param reference_locust_location: Location of the reference locust.
    :param n: Number of closest locusts to find.
    :param movement_points: Points representing locust locations.
    :param max_locusts: Maximum number of locusts in the frame.
    :return: List of indices of the n closest locusts.
    """

    closest_locusts = []
    max_allowable_merge_split_dist = 50

    # Calculate distances and get the indices sorted in ascending order
    dist_order = LA.norm(movement_points[:, 0, :] - reference_locust_location, axis=1)
    ind_order = np.argsort(dist_order)

    # Find the n closest locusts within the maximum locust limit
    for j in range(n):
        if j + 1 < len(ind_order):
            n_closest_locust = ind_order[j + 1]
            if dist_order[j + 1] < max_allowable_merge_split_dist and n_closest_locust < max_locusts:
                closest_locusts.append(n_closest_locust)
        else:
            closest_locusts.append(-1)

    return closest_locusts

def find_closest_n_locusts_in_frame_m(reference_locust_location, n, movement_points, max_locusts):
    """
    Find the n closest locusts to a reference locust's location.

    :param reference_locust_location: Location of the reference locust.
    :param n: Number of closest locusts to find.
    :param movement_points: Points representing locust locations.
    :param max_locusts: Maximum number of locusts in the frame.
    :return: List of indices of the n closest locusts.
    """

    closest_locusts = []
    max_allowable_merge_split_dist = 50

    # Calculate distances and get the indices sorted in ascending order
    dist_order = LA.norm(movement_points - reference_locust_location, axis=1)
    ind_order = np.argsort(dist_order)
    for j in range(n):
        if j + 1 < len(ind_order):
            n_closest_locust = ind_order[j]  # sorts at an ascending order
            if dist_order[n_closest_locust] < max_allowable_merge_split_dist and n_closest_locust < max_locusts:
                closest_locusts.append(n_closest_locust)
            else:
                closest_locusts.append(-1)

        else:
            closest_locusts.append(-1)

    return closest_locusts

def create_white_area_list(area_type_im):
    """
    Create a list of coordinates corresponding to white areas in an image that coreespond to locations
    that locusts can't traverse.

    :param area_type_im: Input image representing different areas.
    :return: List of coordinates for white areas.
    """
    # Extract color channels
    area_type_im_b = area_type_im[:, :, 0]
    area_type_im_g = area_type_im[:, :, 1]
    area_type_im_r = area_type_im[:, :, 2]

    # Multiply color channels to identify white areas
    area_type_mul = area_type_im_r * area_type_im_g * area_type_im_b

    # Identify white pixels
    area_type_white = np.zeros_like(area_type_im_r)
    area_type_white[area_type_mul == 255] = 255
    white_area_list = np.dstack((np.where(area_type_white == 255)[1], np.where(area_type_white == 255)[0]))[0]

    return white_area_list


def calculate_trajectories(video_path, background_im, scale, max_movement, max_stationary, min_distance, max_locusts,
                           min_y, max_y,min_x,max_x, debug_movie_flag, city_name, area_type_im):
    """
       Calculate locust trajectories from a video stream.

       Parameters:
       - video_path (str): Path to the video file.
       - background_im (numpy.ndarray): Background image for comparison.
       - scale (float): Scaling factor for resizing frames.
       - max_movement (float): Maximum allowed movement between frames.
       - max_stationary (float): Maximum allowed stationary distance.
       - min_distance (float): Minimum distance for locust separation.
       - max_locusts (int): Maximum number of locusts to track.
       - min_y, max_y, min_x, max_x (int): Region of interest (ROI) coordinates.
       - debug_movie_flag (bool): Flag to enable debugging videos.
       - city_name (str): Name of the city for processing.
       - area_type_im (numpy.ndarray): Image representing area types.

       Returns:
       - trajlist (list): List of locust trajectories.
       - framelist (list): List of frame indices corresponding to trajectories.
       - frame_counter (int): Total number of processed frames.
       - success (bool): True if the processing is successful, False otherwise.
       - is_cluster (list): List indicating whether locusts are part of a cluster.
       - number_of_locusts (list): List of the number of locusts in each frame.
       - merged_locust_ids (numpy.ndarray): Array indicating merged locust IDs.
       - cluster_origin_id (numpy.ndarray): Array indicating the origin of locust clusters.
       - frame_rate (float): Frames per second in the video.
       - video_length (int): Total number of frames in the video.
       """

    #radius = 21
    #radius = 15
    radius = 15
    diff_heading = 3

    cap = cv2.VideoCapture(video_path)

    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    if int(major_ver) < 3:
        frame_rate = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    else:
        frame_rate = cap.get(cv2.CAP_PROP_FPS)

    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


    heading_array = np.array([])
    trajlist = []
    framelist = []

    is_cluster = []
    number_of_locusts = []
    merged_locust_ids = [[[] for i in range(video_length)] for j in range(max_locusts)] #temporal information
    cluster_origin_id = [[[] for i in range(video_length)] for j in range(max_locusts)] #temporal information


    frame_counter = 0



    first_frame_flag = True
    # prev_points are the centers of the blobs
    accumulated_image = np.zeros_like(background_im)
    h, w = background_im.shape
    location_time_counter = [[[] for i in range(w)] for j in range(h)]
    location_center_time_counter = [[[] for i in range(h)] for j in range(w)]
    ret, background_im = cv2.threshold(background_im, 245, 255, cv2.THRESH_TOZERO_INV)
    background_im = background_im.astype(np.uint8)

    white_area_list = create_white_area_list(area_type_im)

    if debug_movie_flag:
        frame_width = int(w)
        frame_height = int(h)
        centroid_movie = cv2.VideoWriter(dir_name + '\Centroid_Trajectory_Video.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                                (frame_width, frame_height), 0)

        blurred_movie = cv2.VideoWriter(dir_name + '\Blobs.avi',
                                         cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                                         (frame_width, frame_height), 0)
    while (cap.isOpened()):
        ret, current_frame = cap.read()
        if ret:
            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)  # foreground image
            current_frame = cv2.resize(current_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            current_frame= current_frame[min_y:max_y, min_x:max_x]
            ret, current_frame = cv2.threshold(current_frame, 245, 255, cv2.THRESH_TOZERO_INV)
            current_frame = current_frame.astype(np.uint8)
            diff_im = cv2.absdiff(current_frame, background_im)
            ret, diff_im = cv2.threshold(diff_im, 150, 255, cv2.THRESH_TOZERO_INV)
            diff_im = diff_im.astype(np.uint8)
            new_features, centroid_im, blurred_im, is_cluster_frame, number_of_locusts_frame  = compute_blobs(diff_im, radius, city_name, white_area_list)
            ret, thresh_blurred = cv2.threshold(blurred_im, 1, 255, cv2.THRESH_BINARY)
            thresh_blurred = thresh_blurred / 255.0
            locations = np.argwhere(thresh_blurred>0)
            for f in range(len(locations)):
                location_time_counter[locations[f][0]][locations[f][1]].append(frame_counter)

            accumulated_image= accumulated_image + thresh_blurred

            for f in range(len(new_features)):
                location_center_time_counter[int(new_features[f][0][0])][int(new_features[f][0][1])].append(frame_counter)

            if debug_movie_flag:
                if centroid_im is not None:

                    centroid_movie.write(centroid_im)
                    blurred_movie.write(blurred_im)

            # need to handle the case where the number of tracked points in the first and second frames is not the same
            if new_features is not None and not first_frame_flag:
                first_frame_flag = False
                D = pairwise_distances(prev_points[:, 0, :], new_features[:, 0, :]).astype(np.float32)
                same_obj_inds = np.where(D <= min_distance)
                reduction_counter = 1
                while len(np.maximum(same_obj_inds[0],same_obj_inds[1]))> max_locusts and reduction_counter< 19:
                    min_distance_temp = min_distance - 0.05 * reduction_counter
                    reduction_counter = reduction_counter + 1
                    print('min_distance_temp ' + str(min_distance_temp))
                    same_obj_inds = np.where(D <= min_distance_temp)
                    if reduction_counter == 19:
                        same_obj_inds = (same_obj_inds[0][0:max_locusts],same_obj_inds[1][0:max_locusts])


                uniquenewpoints = np.setdiff1d(range(0, D.shape[1]), same_obj_inds[1])
                new_points = new_features[uniquenewpoints, :, :]
                old_points_moved = new_features[same_obj_inds[1], :, :]
                prev_points = prev_points[same_obj_inds[0], :, :]
                curr_points = old_points_moved


                current_movement_points = curr_points
                previous_movement_points = prev_points
                unassigned_points = current_movement_points
                is_cluster_frame =  is_cluster_frame[same_obj_inds[1]] #after this filtering is performed
                # is_cluster_frame contains same number of indices as the current blob centers
                number_of_locusts_frame = number_of_locusts_frame[same_obj_inds[1]]
                if frame_counter == 1:
                    for i in range(previous_movement_points.shape[0]):
                        number_of_locusts.append([number_of_locusts_frame[i]])
                        trajlist.append([current_movement_points[i, 0, :], previous_movement_points[i, 0, :]])
                        is_cluster.append([is_cluster_frame[i]])
                        framelist.append([frame_counter])
                        # if number_of_locusts_frame[i] > 1:
                        #     for v in range(number_of_locusts_frame[i] -2):
                        #         number_of_locusts.append([1])
                        #         trajlist.append([current_movement_points[i, 0, :], previous_movement_points[i, 0, :]])
                        #         is_cluster.append([0])
                        #         framelist.append([frame_counter])
                else:
                    # need to add to existing tracked objects the track, if a new object was added to the tracked object list need to open for it a new track
                    # If there was a detection in the previous frame and you detect a locust in a close by neighborhood of this point
                    # in the next 60 frames than connect this point to the previous's locust trajectory
                    if (len(trajlist) == previous_movement_points.shape[0]):  # the number of tracked locust stayed the same from the previous frame
                        last = []
                        for i in range(len(trajlist)):
                            last.append(trajlist[i][-1])

                        # find for each of the new points in good_new the closest point from the last points of all tracks. If the distance between this point and any of the points is small than "Max_movement" pixels connect them
                        closest_inds = []
                        closest_dists = []
                        for i in range(current_movement_points.shape[0]):
                            closest_inds.append(np.argmin(LA.norm(current_movement_points[i, 0, :] - last, axis=1)))
                            closest_dists.append(np.min(LA.norm(current_movement_points[i, 0, :] - last, axis=1)))


                        for i in range(current_movement_points.shape[0]):
                            current_heading, prev_heading = compute_heading(current_movement_points, trajlist, closest_inds, i)
                            heading_array= np.append(heading_array, np.abs(current_heading - prev_heading))
                            if np.abs(current_heading - prev_heading) < diff_heading:
                                trajlist, framelist, is_cluster, number_of_locusts, unassigned_points  = add_points_to_trajectory\
                                    (trajlist, framelist, is_cluster, number_of_locusts, max_movement,
                                     max_stationary, closest_inds, i, frame_counter, current_movement_points,
                                     closest_dists, previous_movement_points, max_locusts, is_cluster_frame,
                                     number_of_locusts_frame, unassigned_points)

                    elif (len(trajlist) < previous_movement_points.shape[0]):  # we have new locust that were detected and we need to assign them tracks
                        # need to assign the locust that were kept to their previous tracks and create a new track for the new ones

                        last = []
                        for i in range(len(trajlist)):
                            last.append(trajlist[i][-1])
                        # find for each of the new points in good_new the closest point from the last points of all tracks. If the distance between this point and any of the points is small than 5 pixels connect them
                        closest_inds = []
                        closest_dists = []
                        for i in range(len(last)):
                            closest_inds.append(np.argmin(LA.norm(current_movement_points[i, 0, :] - last, axis=1)))
                            closest_dists.append(np.min(LA.norm(current_movement_points[i, 0, :] - last, axis=1)))

                        #new_locust_inds = np.setdiff1d(list(range(0, previous_movement_points.shape[0])), closest_inds)
                        # create all the indices to correspond to new locust and then add them to the list

                        for i in range(len(last)):
                            current_heading, prev_heading = compute_heading(current_movement_points, trajlist, closest_inds, i)
                            heading_array= np.append(heading_array, np.abs(current_heading - prev_heading))
                            if np.abs(current_heading - prev_heading) < diff_heading:
                                trajlist, framelist, is_cluster, number_of_locusts, unassigned_points = \
                                    add_points_to_trajectory(trajlist, framelist, is_cluster, number_of_locusts,
                                     max_movement, max_stationary, closest_inds, i, frame_counter,
                                     current_movement_points, closest_dists, previous_movement_points, max_locusts,
                                                             is_cluster_frame, number_of_locusts_frame, unassigned_points)

                    else:
                        # have more trajectories than currently detected locust
                        # need to assign the detected locust to the closest trajectory or if the distance is too large create a new track
                        last = []
                        for i in range(len(trajlist)):
                            last.append(trajlist[i][-1])
                        # find for each of the new points in good_new the closest point from the last points of all tracks. If the distance between this point and any of the points is small than 5 pixels connect them

                        closest_inds = []
                        closest_dists = []
                        for i in range(current_movement_points.shape[0]):
                            closest_inds.append(np.argmin(LA.norm(current_movement_points[i, 0, :] - last, axis=1)))
                            closest_dists.append(np.min(LA.norm(current_movement_points[i, 0, :] - last, axis=1)))

                        for i in range(current_movement_points.shape[0]):
                            current_heading, prev_heading = compute_heading(current_movement_points, trajlist, closest_inds,i)
                            heading_array= np.append(heading_array, np.abs(current_heading - prev_heading))
                            if np.abs(current_heading - prev_heading) < diff_heading:
                                trajlist, framelist, is_cluster, number_of_locusts, unassigned_points = \
                                    add_points_to_trajectory(trajlist, framelist, is_cluster, number_of_locusts,
                                    max_movement, max_stationary, closest_inds, i, frame_counter,
                                    current_movement_points, closest_dists, previous_movement_points, max_locusts,
                                                             is_cluster_frame, number_of_locusts_frame, unassigned_points)

                    for locust_counter in range(len(number_of_locusts)):
                        locust = number_of_locusts[locust_counter]




                    #we are dealing with a locust cluster, check if in the previous frame the cluster had
                        # more or less locust. If it had the same number of locusts then copy the IDs of the
                        # previous locusts to this frame as well.
                        if len(locust)>1:
                            if locust[-1] == 1 and locust[-2]> 1:
                                #the cluster in no longer a cluster, therefore assign to the locusts that left the cluster, the cluster id

                                n = locust[-2] - locust[-1]  # number of locusts that left the cluster

                                # need to remove n locusts from the merged_locust_ids.
                                # These are the n closest locusts to the current locust in the detected frame
                                reference_locust_location = trajlist[locust_counter][-1]
                                closest_locusts = find_closest_n_locusts_in_frame(
                                    reference_locust_location, n,
                                    current_movement_points, max_locusts)
                                for seprated_locust_index in closest_locusts:
                                    if seprated_locust_index > -1:  # separate only valid locusts
                                        print('seprated_locust_index'+ str(seprated_locust_index))
                                        cluster_origin_id[seprated_locust_index][frame_counter].append(locust_counter)
                                        merged_locust_ids[locust_counter][frame_counter].append(-1)  # meaning it is a single locust


                                # for g in range(len(merged_locust_ids[locust_counter])):
                                #     if merged_locust_ids[locust_counter] != locust_counter:
                                #         #if this is not the "main" locust that forms the cluster
                                #         cluster_origin_id[merged_locust_ids[locust_counter]][frame_counter] = locust_counter

                            elif locust[-1] > 1: # The locust was a cluster in the previous time it was detected and it is still a cluster in this frame
                                if locust[-1] > locust[-2]:
                                    # in the previous frame this cluster was detected and the cluster had less locusts
                                    # This implies that a locust was merged into this cluster, therefore add the merged
                                    # locust into the list of merged locusts in the cluster.

                                    #find the closest locust in the previous frame and assume it is the one that is merged
                                    #closest_inds and closest_dists hold this information.

                                    n = locust[-1] - locust[-2] #number of locusts that were merged
                                    reference_locust_location= trajlist[locust_counter][-1]
                                    closest_locusts = find_closest_n_locusts_in_frame(
                                        reference_locust_location, n,
                                        previous_movement_points, max_locusts)

                                    #merged_locust_indices in the previous frame have to be the n closest locusts that are not the cluster's main locust.
                                    for merged_locust_index in closest_locusts:
                                        if merged_locust_index> -1: # merge only valid locusts
                                            merged_locust_ids[locust_counter][frame_counter].append(merged_locust_index)


                                elif locust[-1] < locust[-2]:
                                    # in the previous frame this cluster was detected and the cluster had more locusts
                                    # This implies that a locust separated from the cluster and we need to remove it
                                    # we also need to update for the separated locust the cluster it originated from

                                    n = locust[-2] - locust[-1]  # number of locusts that left the cluster

                                    # need to remove n locusts from the merged_locust_ids.
                                    # These are the n closest locusts to the current locust in the detected frame
                                    reference_locust_location = trajlist[locust_counter][-1]
                                    closest_locusts = find_closest_n_locusts_in_frame(
                                        reference_locust_location, n,
                                        current_movement_points, max_locusts)
                                    try:
                                        if len(merged_locust_ids) and len(merged_locust_ids[locust_counter])>0:
                                           #closest_locusts are locusts that need to be removed from the cluster
                                            prev_locusts_in_cluster = merged_locust_ids[locust_counter][-1]

                                            for prev_locust_in_cluster in prev_locusts_in_cluster: #add all the locust that did not separate from the cluster
                                                if prev_locust_in_cluster not in closest_locusts:
                                                    merged_locust_ids[locust_counter][frame_counter].append(
                                                        prev_locust_in_cluster)

                                            #need to update cluster_origin_id for separated locusts and indicate they were
                                            # separated from locust cluster
                                            for seprated_locust_index in closest_locusts:
                                                if seprated_locust_index > -1: # separate only valid locusts
                                                    cluster_origin_id[seprated_locust_index][frame_counter].append(locust_counter)
                                    except:
                                        print('error in cluster association')
                                else:
                                    # in the previous frame this cluster was detected the cluster had the same number of locusts
                                    try:
                                        if len(merged_locust_ids[locust_counter][-1]) > 0:
                                            print(len(merged_locust_ids[locust_counter][-1]))
                                        merged_locust_ids[locust_counter][frame_counter] = merged_locust_ids[locust_counter][-1]
                                    except:
                                        print('error in cluster association')

                # Now update the previous frame and previous points

                # Update background model

                if frame_counter > 1 and len(unassigned_points) > 0:
                    locusts_in_prev_frame = []
                    locust_locations_prev_frame = []
                    for q in range(len(trajlist)):
                        locusts_in_prev_frame.append(q)
                        locust_locations_prev_frame.append(trajlist[q][-1])
                    for ind_u in unassigned_points:
                        closest_locust = find_closest_n_locusts_in_frame_m(ind_u, 1, locust_locations_prev_frame, max_locusts)[0]
                        if closest_locust >=0:
                            prev_frame_close_locust_ind = locusts_in_prev_frame[closest_locust]
                            if prev_frame_close_locust_ind< max_locusts:
                                trajlist[prev_frame_close_locust_ind].append(ind_u[0])
                                framelist[prev_frame_close_locust_ind].append(frame_counter)
                                is_cluster[prev_frame_close_locust_ind].append(0)
                                number_of_locusts[prev_frame_close_locust_ind].append(1)
                prev_points = current_movement_points.reshape(-1, 1, 2)
                prev_points = np.concatenate((prev_points, new_points),
                                             axis=0)  # add new points to be tracked that are not close to the previous points
                prev_frame_gray = current_frame.copy()
                frame_counter = frame_counter + 1
                print(frame_counter)
            elif first_frame_flag and new_features is not None:
                prev_points = new_features
                frame_counter = frame_counter + 1
                first_frame_flag = False
        else:
            break

    if debug_movie_flag:
        centroid_movie.release()
        blurred_movie.release()
    save_heatmaps_info(location_time_counter, accumulated_image, location_center_time_counter, dir_name, frame_counter)

    return trajlist, framelist, frame_counter, True, is_cluster, number_of_locusts, \
           merged_locust_ids, cluster_origin_id, frame_rate, video_length

def process_trajectories(trajlist, framelist, Movie_length):
    """
    Process locust trajectories by filtering out short tracks.

    Parameters:
    - trajlist (list): List of locust trajectories.
    - framelist (list): List of frame indices corresponding to trajectories.
    - Movie_length (int): Considered length of a valid trajectory.

    Returns:
    - final_trajectories (list): Filtered list of locust trajectories.
    - final_framelist (list): Filtered list of frame indices corresponding to trajectories.
    """
    final_trajectories = []
    final_framelist = []

    for i in range(len(trajlist)):
        if len(trajlist[i]) > Movie_length:
            final_trajectories.append(trajlist[i])
            final_framelist.append(framelist[i])

    return final_trajectories, final_framelist

def draw_trajectories(background_im, dir_name, scale, video_path, traj_list, movie_length,
                      min_y, max_y, min_x, max_x, fade_time_mask, frame_rate):
    """
    Main function for drawing locust trajectories on the image.

    Parameters:
    - background_im: The background image.
    - dir_name: The directory to save the output video.
    - scale: Scaling factor for the video.
    - video_path: Path to the input video.
    - traj_list: List of locust trajectories.
    - movie_length: Total frames in the video.
    - min_y, max_y, min_x, max_x: Crop boundaries for the video.
    - fade_time_mask: Number of frames to fade the trajectories.
    - frame_rate: Frames per second for the output video.

    Returns:
    None

    This function reads a video, draws locust trajectories on each frame, and
    saves the result as a new video in the specified directory.
    """
    if len(traj_list) > 0:
        font = cv2.FONT_HERSHEY_SIMPLEX  # font type
        fontScale = 0.3  # fontScale
        thickness = 1  # in pixels
        cap = cv2.VideoCapture(video_path)

        cv2.namedWindow("LocusTracker", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("LocusTracker", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        h, w = background_im.shape
        frame_width = int(w * (1 / scale))
        frame_height = int(h * (1 / scale))
        skip_between_displayed_frames = 1

        # Generate random colors for the locust trajectories
        traj_color = np.random.randint(0, 255, (1000, 3))

        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
            print("Directory ", dir_name, " Created ")
        else:
            print("Directory ", dir_name, " already exists")

        # Define the codec and create VideoWriter object.
        # The output is stored in 'dirName+'\Locust_Trajectory_Video.avi'
        out = cv2.VideoWriter(dir_name + '\Locust_Trajectory_Video.avi',
                              cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), int(frame_rate),
                              (frame_width, frame_height))

        # Create mask for trajectory drawing
        frame_counter = 1
        traj_mask = np.zeros((h, w, 3), dtype=np.uint8)
        traj_mask_list = np.zeros((fade_time_mask, h, w, 3), dtype=np.uint8)

        while cap.isOpened():
            ret, current_frame = cap.read()
            current_frame = cv2.resize(current_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            current_frame = current_frame[min_y:max_y, min_x:max_x, :]
            traj_mask_single_im = np.zeros((h, w, 3), dtype=np.uint8)
            if frame_counter < movie_length - 1:
                for j in range(len(traj_list)):
                    if frame_counter in traj_list[j]['Frame_Number'].values:
                        ind = list(traj_list[j]['Frame_Number'].values).index(frame_counter)
                        if ind > 0:
                            x_curr = (traj_list[j]['Location_X'].values[ind] * scale).astype(np.int)
                            y_curr = (traj_list[j]['Location_Y'].values[ind] * scale).astype(np.int)

                            x_prev = (traj_list[j]['Location_X'].values[ind - 1] * scale).astype(np.int)
                            y_prev = (traj_list[j]['Location_Y'].values[ind - 1] * scale).astype(np.int)

                            traj_mask = cv2.line(traj_mask, (x_curr, y_curr), (x_prev, y_prev), traj_color[j].tolist(), 2)
                            traj_mask_single_im = cv2.line(traj_mask_single_im, (x_curr, y_curr), (x_prev, y_prev),
                                                           traj_color[j].tolist(), 2)

                            angle = np.rad2deg(math.atan2(x_curr - x_prev, y_curr - y_prev))
                            if angle == 180:
                                angle = 0

                            current_frame = cv2.circle(current_frame, (x_prev, y_prev), 2, traj_color[j].tolist(), -1)
                            current_frame = cv2.putText(current_frame, 'Locust ' + str(traj_list[j]['Locust_ID'].values[0]),
                                                        (x_prev, y_prev), font, fontScale, traj_color[j].tolist(), thickness,
                                                        cv2.LINE_AA)

                current_frame = cv2.putText(current_frame, 'Frame ' + str(frame_counter), (5, 10), font, fontScale,
                                            traj_color[j].tolist(), thickness, cv2.LINE_AA)

                if skip_between_displayed_frames == 1:
                    if frame_counter < fade_time_mask + 2:
                        traj_mask_disp = traj_mask
                        traj_mask_list[frame_counter - 2, :, :, :] = traj_mask_single_im
                    else:
                        traj_mask_list[np.mod(frame_counter - 2, fade_time_mask), :, :, :] = traj_mask_single_im
                        traj_mask = np.sum(traj_mask_list, axis=0).astype(np.uint8)
                        traj_mask_disp = traj_mask
                        traj_mask = np.zeros((h, w, 3), dtype=np.uint8)

                    img = cv2.add(current_frame, traj_mask_disp)
                else:
                    img = current_frame

                img = cv2.resize(img, None, fx=(1 / scale), fy=(1 / scale), interpolation=cv2.INTER_AREA).astype(np.uint8)
                out.write(img)
                cv2.imshow('LocusTracker', img)

                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    break

                frame_counter += skip_between_displayed_frames
                print(frame_counter)
            else:
                break
        cap.release()
        cv2.destroyAllWindows()
        out.release()
    else:
        print('No Locust detected, consider adjusting the detection parameters')

def get_number_of_measurements_for_movement_check(framelist, i, frame_rate, req_time=1):
    """
    Get the indices for a time window around a specific frame, corresponding to a specified duration.

    Parameters:
    - framelist (list): List of frame numbers.
    - i (int): Index of the frame for which the time window is calculated.
    - frame_rate (float): Frames per second of the video.
    - req_time (int, optional): Required time duration in seconds. Default is 1 second.

    Returns:
    Tuple: A tuple containing the starting and ending indices of the time window.

    This function calculates the starting and ending indices in the `framelist` that
    correspond to the time window around the frame at index `i` with a duration of `req_time` seconds.
    """
    min_time = framelist[i] * (1 / frame_rate) - req_time
    max_time = framelist[i] * (1 / frame_rate)

    min_ind = np.argwhere(framelist[0:i + 1] * np.ones(len(framelist[0:i + 1])) * (1 / frame_rate) - min_time >= 0)[0]
    max_ind = np.argwhere(framelist[0:i + 1] * np.ones(len(framelist[0:i + 1])) * (1 / frame_rate) - max_time >= 0)[0]

    if (len(min_ind) == 0) or (len(max_ind) == 0):
        min_ind = 0
        max_ind = 0

    return min_ind[0], max_ind[0]

def save_combined_data(dir_name, files_in_folder, combined_data_file_path):
    """
    Combine trajectory data from multiple files and save the merged data to a new Excel file.

    Parameters:
    - dir_name (str): Directory path where trajectory files are located.
    - files_in_folder (list): List of trajectory file names in the specified directory.
    - combined_data_file_path (str): File path to save the combined trajectory data.

    Returns:
    pd.DataFrame: Merged trajectory data.

    This function reads trajectory data from individual Excel files in the specified directory (`dir_name`).
    It then combines the data from these files, taking every third row, and saves the merged data to a new Excel file
    specified by `combined_data_file_path`. The merged trajectory data is also returned as a pandas DataFrame.
    """
    merged_traj_file = pd.DataFrame()
    for i in range(len(files_in_folder)):
        traj_file = pd.read_excel(dir_name + '/' + files_in_folder[i])
        if i == 0 and len(traj_file) > 0:
            merged_traj_file = traj_file.iloc[::3, :]
        elif i > 0 and len(traj_file) > 0:
            merged_traj_file = merged_traj_file.append(traj_file.iloc[::3, :])

    merged_traj_file.to_excel(combined_data_file_path)
    return merged_traj_file

def create_area_type_map(area_type_im):
    """
    Process an input area type image to create a new image and calculate the percentage of different colored areas.

    Parameters:
    - area_type_im (numpy.ndarray): Input area type image.

    Returns:
    - numpy.ndarray: New processed image.
    - numpy.ndarray: Binary mask representing the black areas.

    This function takes an input area type image, processes it to create a new image, and calculates the percentage of
    different colored areas (black, white, green, red, and blue). The processed image and a binary mask representing the
    black areas are returned.

    Note:
    - The input image should have three channels (BGR format).
    - Black areas are identified based on the absence of any color (RGB values are all 0).
    - White areas are calculated based on the product of RGB values.
    - Red, green, and blue areas are identified based on specific RGB values.
    - The new image is created by applying convolution and thresholding.

    The function prints the percentage of each colored area and the total sum of areas.
    """

    area_type_im_b = area_type_im[:, :, 0]
    area_type_im_g = area_type_im[:, :, 1]
    area_type_im_r = area_type_im[:, :, 2]

    area_type_black = np.zeros_like(area_type_im_r)
    area_type_black[(area_type_im_b == 0) & (area_type_im_g == 0) & (area_type_im_r == 0)] = 255
    black_area = np.sum((area_type_im_b == 0) & (area_type_im_g == 0) & (area_type_im_r == 0)) / (
                area_type_im_r.shape[0] * area_type_im_r.shape[1]) * 100

    area_type_mul = area_type_im_r * area_type_im_g * area_type_im_b
    white_area = np.sum(area_type_mul) / (
            area_type_im_r.shape[0] * area_type_im_r.shape[1]) * (100 / 255)
    # # turn all white areas to black

    area_type_im_b[area_type_mul == 255] = 0
    area_type_im_g[area_type_mul == 255] = 0
    area_type_im_r[area_type_mul == 255] = 0

    area_type_im_r[(area_type_im_r == area_type_im_g) | (area_type_im_r == area_type_im_b)] = 0
    area_type_im_b[(area_type_im_b == area_type_im_r) | (area_type_im_b == area_type_im_g)] = 0
    area_type_im_g[(area_type_im_g == area_type_im_r) | (area_type_im_g == area_type_im_b)] = 0





    area_type_im_g[(area_type_im_r == 255) & (area_type_im_g == 255)] = 0

    area_type_im_g[(area_type_im_r == 255)] = 255

    area_type_im_g[(area_type_im_g == 176)] = 0

    area_type_im_b[area_type_im_b == 240] = 255

    area_type_im_r[(area_type_im_r == 255)] = 0

    area_type_im_r[(area_type_im_r == 192)] = 255

    area_type_im_b[area_type_im_b < 255] = 0
    area_type_im_g[area_type_im_g < 255] = 0
    area_type_im_r[area_type_im_r < 255] = 0

    tmp = scipy.ndimage.convolve(area_type_im_b / 255, np.ones((3, 3)), mode='constant')
    area_type_im_b = np.logical_and(tmp >= 2, area_type_im_b / 255).astype(np.float32) * 255.0
    tmp = scipy.ndimage.convolve(area_type_im_g / 255, np.ones((3, 3)), mode='constant')
    area_type_im_g = np.logical_and(tmp >= 2, area_type_im_g / 255).astype(np.float32) * 255.0
    tmp = scipy.ndimage.convolve(area_type_im_r / 255, np.ones((3, 3)), mode='constant')
    area_type_im_r = np.logical_and(tmp >= 2, area_type_im_r / 255).astype(np.float32) * 255.0

    red_area = np.sum(area_type_im_r) / (
            area_type_im_r.shape[0] * area_type_im_r.shape[1]) * (100 / 255)
    blue_area = np.sum(area_type_im_b) / (
            area_type_im_r.shape[0] * area_type_im_r.shape[1]) * (100 / 255)
    green_area = np.sum(area_type_im_g) / (
            area_type_im_r.shape[0] * area_type_im_r.shape[1]) * (100 / 255)
    sum= white_area + black_area + green_area + red_area + blue_area
    new_im = np.dstack((area_type_im_b, area_type_im_g, area_type_im_r))
    return new_im, area_type_black

def display_full_locust_trajectories(dir_path, background_im_path):
    """
    Display full locust trajectories based on input trajectory files and save images.

    Parameters:
    - dir_path (str): Directory path containing locust trajectory files (Excel format).
    - background_im_path (str): Path to the background image on which trajectories will be drawn.

    This function reads locust trajectory files from the specified directory, processes the trajectories,
    and draws them on the given background image. The resulting images are saved in a subdirectory named
    'trajectory_images' within the input directory.

    Each trajectory is drawn with a red line connecting consecutive points, and the images are saved in
    PNG format with filenames indicating the locust number.

    Example:
    - If 'locust_file_01.xlsx' and 'locust_file_02.xlsx' are present in the input directory, the resulting
     images will be saved as 'trajectory_locust_01.png' and 'trajectory_locust_02.png' in the 'trajectory_images'
     subdirectory.

    Note:
    - Input trajectory files are assumed to be in Excel format with columns 'Location_X' and 'Location_Y'.
    - The scale factor is set to 0.25 for resizing locust positions.
    """
    files_in_folder = os.listdir(dir_path)
    files_in_folder = [f for f in files_in_folder if (f.startswith('locust_file') and f.endswith('.xlsx'))]

    traj_im_dir = dir_path + '/trajectory_images'
    if not os.path.exists(traj_im_dir):
        os.mkdir(traj_im_dir)

    for j in range(len(files_in_folder)):
        excel_path = dir_path + '/' + files_in_folder[j]
        locust_number = files_in_folder[j].split('.xlsx')[0].split('locust_file_')[1]
        traj_file = pd.read_excel(excel_path)
        location_x = traj_file['Location_X'].values
        location_y = traj_file['Location_Y'].values
        traj_color = np.array([255,0,0])
        scale = 0.25
        background_im = cv2.imread(background_im_path)
        for i in range(len(location_x)- 1):
            x_curr = (location_x[i+1] * scale).astype(np.int)
            y_curr = (location_y[i+1] * scale).astype(np.int)
            x_prev = (location_x[i] * scale).astype(np.int)
            y_prev = (location_y[i] * scale).astype(np.int)
            background_im = cv2.line(background_im, (x_curr, y_curr), (x_prev, y_prev), traj_color.tolist(),2)
        cv2.imwrite(traj_im_dir + '/' + 'trajectory_locust_' + str(locust_number).zfill(2) +'.png', background_im)

def detect_marching_intervals(dir_path, march_distance_thresh, march_time_thresh, frame_rate, pixels_2_cm_background_im, max_locusts):
    """
    Detects intervals where locusts are marching together based on given criteria.

    Parameters:
    - dir_path (str): Path to the directory containing locust trajectory files.
    - march_distance_thresh (float): Threshold distance for considering locusts to be marching together (in cm).
    - march_time_thresh (float): Threshold time for considering locusts to be marching together (in seconds).
    - frame_rate (float): Video frame rate (frames per second).
    - pixels_2_cm_background_im (float): Conversion factor from pixels to centimeters for background image.
    - max_locusts (int): Maximum number of locusts in the dataset.

    Returns:
    None (Modifies locust trajectory files with added 'Marching_Together' column).
    """

    files_in_folder = os.listdir(dir_path)
    files_in_folder = [f for f in files_in_folder if (f.startswith('locust_file') and f.endswith('.xlsx'))]
    data_files = []

    percentage_close_dists_th = 0.8
    percentage_is_moving_th = 0.05
    for j in range(len(files_in_folder)):
        excel_path = dir_path + '/' + files_in_folder[j]
        traj_file = pd.read_excel(excel_path)
        data_files.append(traj_file)

    # is_marching will have the length of the samples of each loucst file

    # generate matrix of checked locust against locust to speed up calculations for pairs that were already computed
    locust_compare_mat = np.zeros((len(data_files), len(data_files)))
    is_marching = [[] for k in range(max_locusts)]

    for i in range(len(data_files)):
        locust_i = data_files[i]
        is_marching_i = [[] for k in range(len(locust_i['Frame_Number'].values))]
        locust_name_i = locust_i['Locust_ID'].values[0]
        is_marching[locust_name_i] = is_marching_i

    for i in range(len(data_files)):
        locust_i = data_files[i]
        is_marching_i = [[] for k in range(len(locust_i['Frame_Number'].values))]
        locust_name_i = locust_i['Locust_ID'].values[0]
        print('Detecting marching intervals of locust ' + str(i))
        for j in range(len(data_files)):
            if j != i and locust_compare_mat[i, j] == 0 and locust_compare_mat[j, i] == 0:
                locust_compare_mat[i, j] = 1
                locust_compare_mat[j, i] = 1
                locust_j = data_files[j]
                locust_name_j = locust_j['Locust_ID'].values[0]
                is_marching_j = [[] for k in range(len(locust_j['Frame_Number'].values))]
                percentage_close_dists_array = np.array([])
                percentage_is_l_1_moving_array = np.array([])
                percentage_is_l_2_moving_array = np.array([])
                same_frame_numbers, locust_1_inds, locust_2_inds = np.intersect1d(locust_i['Frame_Number'].values,
                                                                                  locust_j['Frame_Number'].values,
                                                                                  return_indices=True)

                # find first time when both are active 5mins and choose the largest
                both_active_ind = np.where(same_frame_numbers > march_time_thresh * frame_rate)[0]
                if len(both_active_ind) > 0:
                    first_ind = both_active_ind[0]
                    same_frame_numbers_cut = same_frame_numbers[first_ind:]
                    locust_1_inds_cut = locust_1_inds[first_ind:]
                    locust_2_inds_cut = locust_2_inds[first_ind:]
                    for m in range(
                            len(same_frame_numbers_cut)):  # iterate only over the times that both were detected if intermidiate frames where not present it won't matter
                        min_ind_locust_1, max_ind_locust_1 = get_number_of_measurements_for_movement_check(
                            locust_i['Frame_Number'].values, locust_1_inds_cut[m], frame_rate, march_time_thresh)
                        min_ind_locust_2, max_ind_locust_2 = get_number_of_measurements_for_movement_check(
                            locust_j['Frame_Number'].values, locust_2_inds_cut[m], frame_rate,
                            march_time_thresh)

                        # we now compare between these indices the distances between locust i and locust j and count in how many of them the distance was smaller then the desired distance threshold

                        l_1_x = locust_i['Location_X'].values[
                            (locust_1_inds > min_ind_locust_1) & (locust_1_inds < max_ind_locust_1) - 1]
                        l_1_y = locust_i['Location_Y'].values[
                            (locust_1_inds > min_ind_locust_1) & (locust_1_inds < max_ind_locust_1) - 1]
                        l_2_x = locust_j['Location_X'].values[
                            (locust_2_inds > min_ind_locust_2) & (locust_2_inds < max_ind_locust_2) - 1]
                        l_2_y = locust_j['Location_Y'].values[
                            (locust_2_inds > min_ind_locust_2) & (locust_2_inds < max_ind_locust_2) - 1]
                        if len(l_1_x) > len(l_2_x):
                            l_1_x = l_1_x[0:len(l_2_x)]
                            l_1_y = l_1_y[0:len(l_2_x)]
                        elif len(l_1_x) < len(l_2_x):
                            l_2_x = l_1_x[0:len(l_1_x)]
                            l_2_y = l_1_y[0:len(l_1_x)]
                        distances_cm = (pixels_2_cm_background_im * np.sqrt(
                            np.power(l_1_x - l_2_x, 2) + np.power(l_1_y - l_2_y, 2)))
                        percentage_close_dists = np.sum(distances_cm < march_distance_thresh) / len(distances_cm)
                        # We now count for locust i and j between these indices the number of frames in which they were considered to be moving
                        is_l2_moving = (locust_j['Is_Moving'].values[min_ind_locust_2:max_ind_locust_2] > 0).astype(int)
                        is_l1_moving = (locust_i['Is_Moving'].values[min_ind_locust_1:max_ind_locust_1] > 0).astype(int)

                        if np.any(is_l2_moving > 0) and np.any(is_l1_moving > 0):
                            min_ind_locust_2 = np.where(is_l2_moving > 0)[0][0] + min_ind_locust_2
                            min_ind_locust_1 = np.where(is_l1_moving > 0)[0][0] + min_ind_locust_1
                            percentage_is_l_1_moving = np.sum(
                                locust_i['Is_Moving'].values[min_ind_locust_1:max_ind_locust_1]) / len(
                                locust_i['Is_Moving'].values[min_ind_locust_1:max_ind_locust_1])
                            percentage_is_l_2_moving = np.sum(
                                locust_j['Is_Moving'].values[min_ind_locust_2:max_ind_locust_2]) / len(
                                locust_j['Is_Moving'].values[min_ind_locust_2:max_ind_locust_2])

                            if percentage_close_dists > percentage_close_dists_th and percentage_is_l_1_moving > percentage_is_moving_th and percentage_is_l_2_moving > percentage_is_moving_th:
                                ind_i = np.where(locust_i['Frame_Number'].values == locust_i['Frame_Number'].values[
                                    locust_1_inds_cut[m]])[0][0]
                                ind_j = np.where(locust_j['Frame_Number'].values == locust_j['Frame_Number'].values[
                                    locust_2_inds_cut[m]])[0][0]
                                is_marching_i[ind_i].append(locust_name_j)
                                is_marching_j[ind_j].append(locust_name_i)
                            percentage_is_l_2_moving_array = np.append(percentage_is_l_2_moving_array,
                                                                       percentage_is_l_2_moving)
                            percentage_is_l_1_moving_array = np.append(percentage_is_l_1_moving_array,
                                                                       percentage_is_l_1_moving)
                            percentage_close_dists_array = np.append(percentage_close_dists_array,
                                                                     percentage_close_dists)
                is_marching[locust_name_i] = is_marching_i
                for k in range(len(is_marching_j)):
                    if len(is_marching_j[k]) > 0:
                        is_marching[locust_name_j][k].append(is_marching_j[k][0])

    # when we get here all the information is in is_marching- so we iterate over the locust files and add the data as an extra column
    for i in range(len(data_files)):
        locust_i = data_files[i]
        locust_name_i = locust_i['Locust_ID'].values[0]
        locust_i.insert(locust_i.shape[1], "Marching_Together", is_marching[locust_name_i])
        locust_i.to_excel(dir_path + '/' + files_in_folder[i])

def save_trajectories_to_file(dir_name, scale, final_trajectories, framelist, is_cluster, number_of_locusts, merged_locust_ids,
                              cluster_origin_id, pixels_2_cm_background_im, frame_rate, movement_size, max_step, area_type_im_orig,
                              corner_map, distance_thresh_from_corner):
    """
    Save locust trajectories to CSV files.
    Parameters:
        dir_name (str): Directory path where the CSV files will be saved.
        scale (float): Scaling factor.
        final_trajectories (list): List of locust trajectories.
        framelist (list): List of frame numbers corresponding to each trajectory.
        is_cluster (list): List indicating if a point is part of a cluster (1) or not (0).
        number_of_locusts (list): List indicating the number of locusts in a cluster.
        merged_locust_ids (list): List specifying merged locust IDs.
        cluster_origin_id (list): List specifying the origin ID of a locust separated from a cluster.
        pixels_2_cm_background_im (float): Conversion factor from pixels to centimeters.
        frame_rate (int): Frame rate of the video.
        movement_size (float): Minimum movement size to consider as motion.
        max_step (float): Maximum step size to consider as valid.
        area_type_im_orig (numpy.ndarray): Original area type image.
        corner_map (numpy.ndarray): Corner map indicating corner locations.
        distance_thresh_from_corner (float): Distance threshold from a corner to be considered 'near'.

    Returns:
        None
    """
    # In here we save all the information to the csv files and the movie file with the overlay of the trajectories

    area_type_im, area_type_black  = create_area_type_map(area_type_im_orig)

    area_type_blue = area_type_im[:,:, 0] #area 1
    area_type_green= area_type_im[:,:, 1] # area 2
    area_type_red = area_type_im[:,:, 2] # area 3





    # other area types will be considered as 0
    corner_list = np.dstack((np.where(corner_map == 255)[1] ,np.where(corner_map == 255)[0]))[0]
    blue_area_list = np.dstack((np.where(area_type_blue == 255)[1], np.where(area_type_blue == 255)[0]))[0]
    green_area_list = np.dstack((np.where(area_type_green == 255)[1], np.where(area_type_green == 255)[0]))[0]
    red_area_list = np.dstack((np.where(area_type_red == 255)[1], np.where(area_type_red == 255)[0]))[0]
    black_area_list = np.dstack((np.where(area_type_black == 255)[1], np.where(area_type_black == 255)[0]))[0]

    print('saving locust trajecory files')
    for j in range(len(final_trajectories)):
        with open(dir_name + '\locust_file_temp' + str(j).zfill(2) + '.csv', mode='w', newline='') as locust_file:
            locust_writer = csv.writer(locust_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            locust_writer.writerow(['Frame_Number', "Locust_ID", 'Location_X', 'Location_Y', "Angle", "Is_Moving", "Speed", "Is_cluster",
                                    "Number_of_Locusts", "Near_Corner", "Area_Type", "Marched_Distance", "Stop_Time_Intervals", "Walk_Time_Intervals", "Marched_Distance_Current_Interval"])
            speed_list = []
            accumlated_distance_locust = 0
            first_time_valid = True
            consecutive_steps = 1
            marched_distance_current_interval = 0
            for i in range(len(final_trajectories[j]) - 1):
                speed = -1
                angle = -1

                if (final_trajectories[j][i][0] > 0) and (
                        final_trajectories[j][i][1] > 0):  # don't output out of frame locations

                    x_loc = final_trajectories[j][i][0] * (1 / scale)
                    y_loc = final_trajectories[j][i][1] * (1 / scale)
                    if i==0:
                        prev_sec_x_loc = x_loc
                        prev_sec_y_loc = y_loc
                    if np.mod(i,frame_rate * 10) == 0 and i >= frame_rate:
                        current_sec_x_loc = x_loc
                        current_sec_y_loc = y_loc
                        movement_pixels = np.sqrt(
                            np.power(current_sec_x_loc - prev_sec_x_loc, 2) + np.power(current_sec_y_loc - prev_sec_y_loc, 2))
                        if movement_pixels>(4 *scale):
                            movement_cm_second = (pixels_2_cm_background_im * movement_pixels)
                            accumlated_distance_locust = accumlated_distance_locust + movement_cm_second  # total distance the locust walked up to this frame
                            marched_distance_current_interval = marched_distance_current_interval + movement_cm_second
                        prev_sec_x_loc = current_sec_x_loc
                        prev_sec_y_loc = current_sec_y_loc
                    location = np.array([x_loc * scale, y_loc * scale])
                    if i< (len(final_trajectories[j]) -2) and (framelist[j][i + 1] != framelist[j][i]):
                        next_x_loc =  final_trajectories[j][i + 1][0] * (1 / scale)
                        next_y_loc = final_trajectories[j][i + 1][1] * (1 / scale)

                        angle = np.rad2deg(math.atan2(next_x_loc - x_loc, next_y_loc - y_loc))
                        if angle == 180:
                            angle = 0
                        movement_pixels_speed = np.sqrt(np.power(x_loc-next_x_loc,2) + np.power(y_loc-next_y_loc,2))
                        if movement_pixels_speed> (2 * scale):
                            movement_cm = (pixels_2_cm_background_im * movement_pixels_speed )
                            speed = movement_cm / ((framelist[j][i + 1] - framelist[j][i]) * (1 / frame_rate))
                            speed_list.append(speed)
                        else:
                            speed_list.append(0)

                    near_corner = 0
                    area_type = 0

                    #need to check here if the current location is less than the distance_thresh_from_corner from any corner

                    dist_order = LA.norm(corner_list - location, axis=1) * pixels_2_cm_background_im
                    ind_order = np.argsort(dist_order)
                    closest_corner = ind_order[0]  # sorts at an ascending order
                    if dist_order[closest_corner] < distance_thresh_from_corner:
                        near_corner = 1
                    if len(np.argwhere((blue_area_list[:,0] == location[0]) &(blue_area_list[:,1] == location[1])))>0:
                        area_type = 1
                    elif len(np.argwhere((green_area_list[:,0] == location[0]) &(green_area_list[:,1] == location[1])))>0:
                        area_type = 3
                    elif len(np.argwhere((red_area_list[:,0] == location[0]) &(red_area_list[:,1] == location[1])))>0:
                        area_type = 2
                    elif len(np.argwhere((black_area_list[:,0] == location[0]) &(black_area_list[:,1] == location[1])))>0 :
                        area_type = 4
                    # do an average of last sec and if this average is above 0.25 cm per second then there is
                    # movement at this time instant. For the boundary put unknown -1 number. For movement put 1, if stationary put 0.
                    is_movement = -1
                    stopped_walking = -1
                    walking = -1
                    average_speed  = 0
                    walk_stop_flag = False

                    # need to account for the case that there will be jumps in the detection times so we must look at framelist[j][i]
                    if (framelist[j][len(speed_list)-1] - framelist[j][0]) >  np.ceil(frame_rate): # need measurements that last a second
                        min_ind, max_ind = get_number_of_measurements_for_movement_check(framelist[j], i, frame_rate)
                        accumlated_distance = np.sum((np.ones(len(speed_list[min_ind: max_ind+1])) * (1 / frame_rate)) * speed_list[min_ind: max_ind+1])
                        average_speed = accumlated_distance # because we are summing only 1 second


                        if average_speed > movement_size: # motion is defined as moving by more than 0.25 cm per second
                            is_movement = 1
                        else:
                            is_movement = 0
                        if first_time_valid:
                            prev_frame_is_movement = is_movement
                            first_time_valid = False
                        else:
                            if prev_frame_is_movement != is_movement:
                                walk_stop_flag = True
                                current_sec_x_loc = x_loc
                                current_sec_y_loc = y_loc
                                movement_pixels = np.sqrt(
                                    np.power(current_sec_x_loc - prev_sec_x_loc, 2) + np.power(
                                        current_sec_y_loc - prev_sec_y_loc, 2))
                                if movement_pixels > (4 * scale):
                                    movement_cm_second = (pixels_2_cm_background_im * movement_pixels)
                                    accumlated_distance_locust = accumlated_distance_locust + movement_cm_second  # total distance the locust walked up to this frame
                                    marched_distance_current_interval = marched_distance_current_interval + movement_cm_second
                                prev_sec_x_loc = current_sec_x_loc
                                prev_sec_y_loc = current_sec_y_loc
                                if prev_frame_is_movement == 0:
                                    stopped_walking = consecutive_steps
                                else:
                                    walking = consecutive_steps
                            else:
                                consecutive_steps = consecutive_steps + 1

                            prev_frame_is_movement = is_movement
                    if average_speed < max_step:
                        if i<(len(final_trajectories[j]) -2) and (framelist[j][i + 1] != framelist[j][i]):
                            locust_writer.writerow([framelist[j][i], j, x_loc, y_loc, angle, is_movement, average_speed, is_cluster[j][i],
                                                    number_of_locusts[j][i], near_corner, area_type, accumlated_distance_locust, stopped_walking, walking,
                                                    marched_distance_current_interval])
                    if walk_stop_flag:
                        consecutive_steps = 1
                        marched_distance_current_interval = 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, default="C:/PycharmProjects/LocusTracker/NY_2min.mp4", help="path to video with movements")
    parser.add_argument('--background_video_path', type=str, default="ny_background_im.jpg", help="path to background video or image")
    parser.add_argument('--city_model_template', type=str, default="ny_template.png",
                        help="path to background video or image")
    parser.add_argument('--area_type_map', type=str, default="ny_street_types.png",
                        help="street coloring map according to width classification")
    parser.add_argument('--corner_map', type=str, default="only_corners_new_york.png",
                        help="corner map to estimate locust decisions")
    parser.add_argument('--reduction_scale', type=float, default=0.25, help="Working scale on the input video")
    parser.add_argument('--max_movement', type=float, default=15,
                        help="Maximum locust possible movement in a single frame")
    parser.add_argument('--max_stationary', type=float, default=20000,
                        help="Maximal number of frames a locust can stay in the same place before opening a new track")
    parser.add_argument('--min_distance', type=float, default=1,
                        help="Minimal Euclidean distance between detected locusts (should correspond to the Locust size )")
    parser.add_argument('--max_locusts', type=int, default=50,
                        help="Maximal number of Locusts in the maze )")
    parser.add_argument('--debug_movie_flag', type=bool, default=False,
                        help="Save useful videos for debugging purposes)")
    parser.add_argument('--fade_time_mask', type=int, default=20,
                        help="Number of frames to display previous locust location in a movie)")
    parser.add_argument('--city_name', type=str, default="new_york",
                        help="Name of the city- new_york, cairo, rome)")
    parser.add_argument('--movement_size', type=float, default=0.25,
                        help="movement threshold (cm/sec). If exceeded then the locust is considered to be moving)")
    parser.add_argument('--max_step', type=float, default=5,
                        help="Maximum movement in (cm/sec) possible. If exceeded discard the prediction. )")
    parser.add_argument('--dist_thresh_from_corner', type=float, default=1,
                        help="Distance threshold to be considered near a_corner (cm).)")
    parser.add_argument('--march_time_thresh', type=float, default=3,
                        help="Time threshold to be considered as marching together (sec).)")
    parser.add_argument('--march_distance_thresh', type=float, default=5,
                        help="Distance threshold to be considered as marching together (cm).)")
    config_params = parser.parse_args()
    video_path = config_params.video_path
    background_video_path = config_params.background_video_path
    scale = config_params.reduction_scale
    max_movement = config_params.max_movement
    min_distance = config_params.min_distance
    max_stationary = config_params.max_stationary
    max_locusts = config_params.max_locusts
    debug_movie_flag = config_params.debug_movie_flag
    fade_time_mask = config_params.fade_time_mask
    city_name = config_params.city_name
    movement_size = config_params.movement_size
    max_step = config_params.max_step
    march_distance_thresh = config_params.march_distance_thresh
    march_time_thresh = config_params.march_time_thresh
    vid_name = video_path.split('/')[-1].split('.')[0]
    city_template_im = cv2.imread(config_params.city_model_template,0)
    area_type_im = cv2.imread(config_params.area_type_map)
    corner_map = cv2.imread(config_params.corner_map,0)
    distance_thresh_from_corner = config_params.dist_thresh_from_corner
    city_template_im = city_template_im/np.max(city_template_im)
    pixels_2_cm_template = 120 / np.mean(city_template_im.shape)  # the size of each pixel in the template image

    if str(background_video_path).__contains__('.mp4'):
        background_vid_name = background_video_path.split('/')[-1].split('.')[0]
        # get movie here
        cap = cv2.VideoCapture(background_video_path)
        cap.set(1, 0)  # read first frame and use it as the background image for background subtraction
        ret, img = cap.read()
        cap.release()
    else:
        img= cv2.imread(background_video_path)
        ret = True
    if ret:
        maze_coordinates = []
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cropped_image = img.copy()
        cv2.imshow("LocusTracker", img)
        print('Click on upper left corner of maze')
        cv2.setMouseCallback("LocusTracker", click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cropped_image, min_y, max_y,min_x,max_x = get_cropped_maze_region(cropped_image, maze_coordinates)
        background_im = cropped_image[:, :, 0]

        area_type_im_resized = cv2.resize(area_type_im, (background_im.shape[1], background_im.shape[0]))
        corner_map_resized = cv2.resize(corner_map, (background_im.shape[1], background_im.shape[0]))
        dir_name = vid_name + '_Outputs'
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print("saving outputs in ", dir_name)
        else:
            now = datetime.now()
            date_and_time = now.strftime("%d_%m_%Y_%H_%M_%S")
            dir_name = vid_name + '_' + date_and_time + '_Outputs'
            print("saving outputs in ", dir_name)
            os.makedirs(dir_name)
        pixels_2_cm_background_im = calculate_active_area(city_template_im, background_im, dir_name)
        #downscale area_type_im and corner_map to dimensions of area of interest

        trajlist, framelist, movie_length, valid_movie, is_cluster, number_of_locusts, \
        merged_locust_ids, cluster_origin_id, frame_rate, video_length = calculate_trajectories(video_path,
                               background_im, scale, max_movement, max_stationary, min_distance, max_locusts,
                               min_y, max_y,min_x,max_x, debug_movie_flag, city_name, area_type_im_resized)

        cv2.imwrite(dir_name + '/background_im.png', background_im)
        if valid_movie:
            save_trajectories_to_file(dir_name, scale, trajlist, framelist, is_cluster, number_of_locusts,
                                      merged_locust_ids, cluster_origin_id, pixels_2_cm_background_im, frame_rate,
                                      movement_size, max_step, area_type_im_resized, corner_map_resized,
                                      distance_thresh_from_corner)

            traj_list = merge_trajectories(dir_name)




            # Delete the old temporary locust files and save only the merged ones
            files_in_folder = os.listdir(dir_name)
            files_in_folder = [f for f in files_in_folder if (f.startswith('locust_file_temp') and f.endswith('.csv'))]
            for f in files_in_folder:
                os.remove(dir_name + '/' + f)

            # here we check if locusts are marching together and save the updated files
            detect_marching_intervals(dir_name, march_distance_thresh, march_time_thresh, frame_rate,
                                      pixels_2_cm_background_im, max_locusts)

            # Save the combined output file

            files_in_folder = os.listdir(dir_name)
            files_in_folder = [f for f in files_in_folder if (f.startswith('locust_file') and f.endswith('.xlsx'))]
            combined_data_file_path = dir_name + '\combined_data.xlsx'
            save_combined_data(dir_name, files_in_folder, combined_data_file_path)

            background_im_path = dir_name + '/background_im.png'
            display_full_locust_trajectories(dir_name, background_im_path)

            draw_trajectories(background_im, dir_name, scale, video_path, traj_list, movie_length, min_y, max_y, min_x,
                               max_x, fade_time_mask, frame_rate)
