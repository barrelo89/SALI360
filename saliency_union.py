import numpy as np
import os
import cv2

def img_concatenate(img_sequence, img_order, axis = 0):

    img = []
    for idx in img_order:
        img.append(img_sequence[idx])

    return np.concatenate(img, axis = axis), img_sequence[2], img_sequence[3]

def img_split(img, axis = 0):

    #split axis: 0(row-based), 1(column-based)
    if len(img.shape) == 3:
        height, width, _ = img.shape
    else:
        height, width = img.shape
    row_axis = int(height / 2)
    column_axis = int(width / 2)

    if axis == 0:
        return img[:row_axis, :], img[row_axis:, :]

    elif axis == 1:
        return img[:, :column_axis], img[:, column_axis:]

    else:
        return img[:row_axis, :], img[row_axis:, :], img[:, :column_axis], img[:, column_axis:]

def rotate_img(img, iteration = 1):
    #0: remain same, +n: rotate left in 90*n degree, -n: rotate right in 90*n degree (some part of img is sliced off)
    if len(img.shape) == 3:
        height, width, _ = img.shape
    else:
        height, width = img.shape

    rotated_img = np.zeros(img.shape, dtype = np.uint8)

    if iteration == -1:

        for idx in range(height):
            rotated_img[:, height-1-idx] = img[idx, :]

    elif iteration == 1:

        for idx in range(height):
            rotated_img[:, idx] = np.flip(img[idx, :], axis = 0)

    elif iteration == 0:

        rotated_img = img

    else:

        for idx in range(height):
            rotated_img[height-1-idx, :] = np.flip(img[idx, :], axis = 0)

    return rotated_img

def inverse(list):

    return [-item for item in list]

def frame_split(frame):
    #return a list of screens in an order of right, left, up, down, front and back
    if len(frame.shape) == 3:
        height, width, _ = frame.shape
    else:
        height, width = frame.shape

    unit_height = int(height / 2)
    unit_width = int(width / 3)

    #1st layer
    right_frame = frame[:unit_height, :unit_width]
    left_frame = frame[:unit_height, unit_width:2*unit_width]
    up_frame = frame[:unit_height, 2*unit_width:]
    #2nd layer
    down_frame = frame[unit_height:, :unit_width]
    front_frame = frame[unit_height:, unit_width:2*unit_width]
    back_frame = frame[unit_height:, 2*unit_width:]

    return [right_frame, left_frame, up_frame, down_frame, front_frame, back_frame]

def cube_img_concatenate(img_sequence, img_order, up_rotation_list, down_rotation_list):

    img, up_img, down_img = img_concatenate(img_sequence, img_order, axis = 1)

    for idx in range(len(up_rotation_list)):

        if idx == 0:
            rotated_up_img = rotate_img(up_img, iteration = up_rotation_list[idx])
            rotated_down_img = rotate_img(down_img, iteration = down_rotation_list[idx])

        else:
            rotated_up_img = np.concatenate((rotated_up_img, rotate_img(up_img, iteration = up_rotation_list[idx])), axis = 1)
            rotated_down_img = np.concatenate((rotated_down_img, rotate_img(down_img, iteration = down_rotation_list[idx])), axis = 1)

    #after testing, using half-size of horizontally split rotated_up, and down img turned out to produce more human-like saliency map
    if len(rotated_up_img.shape) == 3:
        height, width, _ = rotated_up_img.shape

    else:
        height, width = rotated_up_img.shape

    flattened_up_img = rotated_up_img[int(height/2):, :]
    flattened_down_img = rotated_down_img[:int(height/2), :]

    flattened_cube_img = np.concatenate([flattened_up_img, img, flattened_down_img], axis = 0)
    #flattened_cube_img = np.concatenate([rotated_up_img, img, rotated_down_img], axis = 0)
    return flattened_cube_img

def up_down_saliency_union(flattened_cube_saliency, up_saliency, down_saliency, up_rotation_list, down_rotation_list):

    inverse_up_rotation_list = inverse(up_rotation_list)
    inverse_down_rotation_list = inverse(down_rotation_list)

    height, width = flattened_cube_saliency.shape

    flattened_up_saliency = flattened_cube_saliency[:int(height/4), :]
    flattened_down_saliency = flattened_cube_saliency[-int(height/4):, :]

    #saliency_union for up and down side
    for idx in range(len(inverse_up_rotation_list)):
        up_input_img = np.zeros(up_saliency.shape)
        down_input_img = np.zeros(up_saliency.shape)

        up_input_img[-int(height/4):, :] = flattened_up_saliency[:, int(width/len(inverse_up_rotation_list))*idx:int(width/len(inverse_up_rotation_list))*(idx+1)]
        down_input_img[:int(height/4):, :] = flattened_down_saliency[:, int(width/len(inverse_up_rotation_list))*idx:int(width/len(inverse_up_rotation_list))*(idx+1)]

        up_rotated_img = rotate_img(up_input_img, inverse_up_rotation_list[idx])
        down_rotated_img = rotate_img(down_input_img, inverse_down_rotation_list[idx])

        up_saliency[np.logical_or(up_saliency, up_rotated_img)] = np.maximum(up_saliency[np.logical_or(up_saliency, up_rotated_img)], up_rotated_img[np.logical_or(up_saliency, up_rotated_img)])
        down_saliency[np.logical_or(down_saliency, down_rotated_img)] = np.maximum(down_saliency[np.logical_or(down_saliency, down_rotated_img)], down_rotated_img[np.logical_or(down_saliency, down_rotated_img)])

    return up_saliency, down_saliency

def argsort(input_list):
    #ascending order (sorted() is also in an ascending order)
    return sorted(range(len(input_list)), key = input_list.__getitem__)

def dist(point_1, point_2):
    return np.sqrt((point_1[0] - point_2[0])**2 + (point_1[1] - point_2[1])**2)

def boundary(row, col, row_idx, col_idx, window_size):
    #up boundary
    if row_idx - window_size < 0:
        up_boundary = 0
    else:
        up_boundary = row_idx - window_size

    #down boundary
    if row_idx + window_size > row - 1:
        down_boundary = row
    else:
        down_boundary = row_idx + window_size + 1

    #left boundary
    if col_idx - window_size < 0:
        left_boundary = 0
    else:
        left_boundary = col_idx - window_size

    #right boundary
    if col_idx + window_size > col - 1:
        right_boundary = col
    else:
        right_boundary = col_idx + window_size + 1

    return up_boundary, down_boundary, left_boundary, right_boundary

def find_max_salient_regions(matrix, sub_square_num, window_size):

    row, col = matrix.shape #1024 x 1024*5
    side_length = int(col / 5)
    cen_row, cen_col = row / 2, col / 2

    column_sum_matrix = np.zeros((row - window_size + 1, col))
    row_sum_matrix = np.zeros((row - window_size + 1, col - window_size + 1, 3))
    dist_matrix = np.zeros((row - window_size + 1, col - window_size + 1))

    for col_idx in range(col):

        column_sum = 0

        for row_idx in range(window_size):
            column_sum += int(matrix[row_idx, col_idx])

        column_sum_matrix[0, col_idx] = column_sum

        for row_idx in range(1, row - window_size + 1):
            column_sum += int(matrix[row_idx + window_size - 1, col_idx]) - int(matrix[row_idx - 1, col_idx])
            column_sum_matrix[row_idx, col_idx] = column_sum

    for row_idx in range(row - window_size + 1):

        row_sum = 0

        for col_idx in range(window_size):
            row_sum += int(column_sum_matrix[row_idx, col_idx])

        row_sum_matrix[row_idx, 0, :] = [row_idx, 0, row_sum]
        dist_matrix[row_idx, 0] = dist([row_idx, 0], [cen_row, cen_col])

        for col_idx in range(1, col - window_size + 1):
            row_sum += int(column_sum_matrix[row_idx, col_idx + window_size - 1]) - int(column_sum_matrix[row_idx, col_idx - 1])
            row_sum_matrix[row_idx, col_idx] = [row_idx, col_idx, row_sum]
            dist_matrix[row_idx, col_idx] = dist([row_idx, col_idx], [cen_row, cen_col])

    #it would be better to nullify the saliency value in the front view in the saliency map itself
    row_sum_matrix[:, int(2*side_length - window_size):int(3*side_length), 2] = 0

    global dist_max, row_sum_max

    dist_max = dist_matrix.max()
    row_sum_max = row_sum_matrix.max()
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------
    #need to modify this line:
    row_sum_matrix[:, :, 2] = (row_sum_matrix[:, :, 2] / row_sum_max)# * (dist_matrix / dist_max)
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------
    max_sum_list = []
    max_sum_coordinate = []

    row, col, _ = row_sum_matrix.shape

    for idx in range(sub_square_num):

        max_row = np.where(row_sum_matrix[:,:, 2] == row_sum_matrix[:,:, 2].max())[0][0]
        max_col = np.where(row_sum_matrix[:,:, 2] == row_sum_matrix[:,:, 2].max())[1][0]

        max_sum = row_sum_matrix[max_row, max_col, 2]
        max_coordinate = [max_row, max_col]

        max_sum_list.append(max_sum)
        max_sum_coordinate.append(max_coordinate)

        up_boundary, down_boundary, left_boundary, right_boundary = boundary(row, col, max_row, max_col, window_size)

        #within the boundary, leave out the row_sum_matrix elements whose distance to max_coordinate is less than the window_size
        for row_coord in range(up_boundary, down_boundary):
            for col_coord in range(left_boundary, right_boundary):
                if dist(max_coordinate, [row_coord, col_coord]) < window_size:
                    row_sum_matrix[row_coord, col_coord, 2] = 0
    #print(row_sum_matrix[:int(side_length/ 2), int(1.5*side_length):int(3*side_length), 2])
    return max_sum_list, max_sum_coordinate

def up_down_salient_region(matrix, sub_square_num, window_size):

    row, col = matrix.shape
    cen_row, cen_col = row + int(row / 2), int(col / 2)

    column_sum_matrix = np.zeros((row - window_size + 1, col))
    row_sum_matrix = np.zeros((row - window_size + 1, col - window_size + 1, 3))
    dist_matrix = np.zeros((row - window_size + 1, col - window_size + 1))

    for col_idx in range(col):

        column_sum = 0

        for row_idx in range(window_size):
            column_sum += int(matrix[row_idx, col_idx])
        column_sum_matrix[0, col_idx] = column_sum

        for row_idx in range(1, row - window_size + 1):
            column_sum += int(matrix[row_idx + window_size - 1, col_idx]) - int(matrix[row_idx - 1, col_idx])
            column_sum_matrix[row_idx, col_idx] = column_sum

    for row_idx in range(row - window_size + 1):

        row_sum = 0

        for col_idx in range(window_size):
            row_sum += int(column_sum_matrix[row_idx, col_idx])

        row_sum_matrix[row_idx, 0, :] = [row_idx, 0, row_sum]
        dist_matrix[row_idx, 0] = dist([row_idx, 0], [cen_row, cen_col])

        for col_idx in range(1, col - window_size + 1):
            row_sum += int(column_sum_matrix[row_idx, col_idx + window_size - 1]) - int(column_sum_matrix[row_idx, col_idx - 1])
            row_sum_matrix[row_idx, col_idx] = [row_idx, col_idx, row_sum]
            dist_matrix[row_idx, col_idx] = dist([row_idx, col_idx], [cen_row, cen_col])

    #dist_max = dist_matrix.max() #using global variable, we can unify the scale of distance and row_sum
    #row_sum_max = row_sum_matrix.max()
    #print(dist_max, row_sum_max)

    row_sum_matrix[:, :, 2] = (row_sum_matrix[:, :, 2] / row_sum_max) * (dist_matrix / dist_max)

    max_sum_list = []
    max_sum_coordinate = []

    row, col, _ = row_sum_matrix.shape

    for idx in range(sub_square_num):

        max_row = np.where(row_sum_matrix[:,:, 2] == row_sum_matrix[:,:, 2].max())[0][0]
        max_col = np.where(row_sum_matrix[:,:, 2] == row_sum_matrix[:,:, 2].max())[1][0]

        max_sum = row_sum_matrix[max_row, max_col, 2]
        max_coordinate = [max_row, max_col]

        max_sum_list.append(max_sum)
        max_sum_coordinate.append(max_coordinate)

        up_boundary, down_boundary, left_boundary, right_boundary = boundary(row, col, max_row, max_col, window_size)

        #within the boundary, leave out the row_sum_matrix elements whose distance to max_coordinate is less than the window_size
        for row_coord in range(up_boundary, down_boundary):
            for col_coord in range(left_boundary, right_boundary):
                if dist(max_coordinate, [row_coord, col_coord]) < window_size:
                    row_sum_matrix[row_coord, col_coord, 2] = 0

    return max_sum_list, max_sum_coordinate


    #max_row = np.where(row_sum_matrix[:,:, 2] == row_sum_matrix[:,:, 2].max())[0][0]
    #max_col = np.where(row_sum_matrix[:,:, 2] == row_sum_matrix[:,:, 2].max())[1][0]

    #max_sum = row_sum_matrix[max_row, max_col, 2]
    #max_coordinate = [max_row, max_col]

    #return max_sum, max_coordinate

def region_name(coordinate_list, boundary_value_list, boundary_name_list, side_length, window_size):

    for idx in range(len(boundary_name_list)):

        #target name done
        if boundary_value_list[idx] <= coordinate_list[1] < boundary_value_list[idx + 1]:
            target_region = boundary_name_list[idx]
            target_address = idx
            target_row = coordinate_list[0]
            target_column = coordinate_list[1] % side_length

    condition = target_column + window_size > side_length - 1

    if condition: #when a region spans over two sequent small frames

        if target_address < len(boundary_name_list) - 1:
            next_region = boundary_name_list[target_address + 1]
        else:
            next_region = boundary_name_list[1]

        next_row = coordinate_list[0]
        next_column = 0
        target_width = side_length - target_column#(coordinate_list[1] % side_length)
        next_width = window_size - target_width
        #print(target_region, target_address, target_row, target_column, target_width, next_region, next_row, next_column, next_width)
        return [target_region, next_region], [target_row, next_row], [target_column, next_column], [target_width, next_width]

    else:
        target_width = window_size
        #print([target_region], [target_row], [target_column], [target_width])
        return [target_region], [target_row], [target_column], [target_width]

def salient_region_selection(img_sequence, max_sum_list, max_sum_coordinate_list, up_sum_list, up_coordinate_list, down_sum_list, down_coordinate_list, flattened_front_view, flattened_front_view_saliency, num_selected_regions):

    dictionary = {}

    for coordinate, max_sum in zip(max_sum_coordinate_list, max_sum_list):
        dictionary[max_sum] = {'F' : coordinate}
    #    cv2.imshow('img', matrix[coordinate[0]: coordinate[0] + window_size, coordinate[1]: coordinate[1] + window_size])
    #    cv2.waitKey(0)

    for up_coordinate, up_sum in zip(up_coordinate_list, up_sum_list):
        dictionary[up_sum] = {'U': up_coordinate}
    #    cv2.imshow('img', up_saliency[up_coordinate[0]: up_coordinate[0] + window_size, up_coordinate[1]: up_coordinate[1] + window_size])
    #    cv2.waitKey(0)

    for down_coordinate, down_sum in zip(down_coordinate_list, down_sum_list):
        dictionary[down_sum] = {'D': down_coordinate}
    #    cv2.imshow('img', down_saliency[down_coordinate[0]: down_coordinate[0] + window_size, down_coordinate[1]: down_coordinate[1] + window_size])
    #    cv2.waitKey(0)

    #max_sum_list vs up_sum_list vs down_sum_list sorting needed
    #keys of the dictionary is the max_sum value
    sorted_by_max_sum_list = sorted(dictionary) #keys sorted in an ascending order
    selected_regions = {}

    #newly added-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if num_selected_regions > len(sorted_by_max_sum_list):
        num_selected_regions = len(sorted_by_max_sum_list)
    #newly added-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    for idx in range(1, num_selected_regions + 1):
        selected_regions[sorted_by_max_sum_list[-idx]] = dictionary[sorted_by_max_sum_list[-idx]]

    #print(selected_regions)
    row, col = flattened_front_view_saliency.shape
    num_small_frames = 5
    side_length = int(col / num_small_frames)
    window_size = int(side_length / 4)
    #############################################salient_patch_size_ratio

    boundary_value_list = np.array([side_length*idx for idx in range(num_small_frames + 1)])
    boundary_name_list = ['B', 'L', 'F', 'R', 'B']

    encoding_info = {}
    segment_list = []
    order = 0
    #
    keys_saliency = []
    values_region_info = []

    for key, value in selected_regions.items():
        keys_saliency.append(key)
        values_region_info.append(value)

    keys_saliency = np.array(keys_saliency)
    values_region_info = np.array(values_region_info)

    desending_sorted_keys_saliency_idx = keys_saliency.argsort()[::-1]

    sorted_keys = keys_saliency[desending_sorted_keys_saliency_idx]
    sorted_values = values_region_info[desending_sorted_keys_saliency_idx]
    #
    for key, value in zip(sorted_keys, sorted_values):
        #ex) selected_regions = {100: {'F': [100, 200]}, 200: {'D':[200, 300]}}
        #up or down이면 이름, 좌표는 별도로 생각할 필요 없이 그대로 유지하면 된다.
        #print(key, value)
        if 'U' in list(value.keys()):
            up_view = img_sequence[2]
            coordinate = list(value.values())[0]
            print(coordinate)
            print(up_view[coordinate[0]:coordinate[0]+window_size, coordinate[1]:coordinate[1]+window_size].shape)

            segment_list.append(up_view[coordinate[0]:coordinate[0]+window_size, coordinate[1]:coordinate[1]+window_size])

            encoding_info[order] = {"row": str(list(value.values())[0][0]), "column": str(list(value.values())[0][1]), "width": str(window_size), "name": str(list(value.keys())[0]), "saliency": str(key)}
            order += 1

        elif 'D' in list(value.keys()):
            down_view = img_sequence[3]
            coordinate = list(value.values())[0]
            print(coordinate)
            print(down_view[coordinate[0]:coordinate[0]+window_size, coordinate[1]:coordinate[1]+window_size].shape)

            segment_list.append(down_view[coordinate[0]:coordinate[0]+window_size, coordinate[1]:coordinate[1]+window_size])

            encoding_info[order] = {"row": str(list(value.values())[0][0]), "column": str(list(value.values())[0][1]), "width": str(window_size), "name": str(list(value.keys())[0]), "saliency": str(key)}
            order += 1

        else:
            coordinate = list(value.values())[0]
            print(coordinate)
            print(flattened_front_view[coordinate[0]:coordinate[0]+window_size, coordinate[1]:coordinate[1]+window_size].shape)

            segment_list.append(flattened_front_view[coordinate[0]:coordinate[0]+window_size, coordinate[1]:coordinate[1]+window_size])

            region, row, col, width = region_name(list(value.values())[0], boundary_value_list, boundary_name_list, side_length, window_size)
            for seg in range(len(region)):
                encoding_info[order] = {"row": str(row[seg]), "column": str(col[seg]), "width": str(width[seg]), "name": str(region[seg]), "saliency": str(key)}
                order += 1

    out_img = []
    img_layer = []
    row_sum = 0
    for idx in range(len(segment_list)):

        _, col, _ = segment_list[idx].shape
        print(segment_list[idx].shape)
        row_sum += col
        img_layer.append(segment_list[idx])

        if row_sum == side_length:
            out_img.append(np.concatenate(img_layer, axis = 1))
            img_layer = []
            row_sum = 0

    out_img = np.concatenate(out_img, axis = 0)

    return out_img, encoding_info

def saliency_encoder(img, front_sub_square_num, up_sub_square_num, down_sub_square_num, num_selected_regions, resize_ratio, salient_patch_size, output):
    #split the cube frame into each small frame
    img_sequence = frame_split(img) #return small frames in the following order of 'frame_order'
    frame_order = ['right', 'left', 'up', 'down', 'front', 'back']

    height, width, _ = img_sequence[0].shape

    #extract the salienc maps of up-frame and down-frame
    #rbd, up_saliency = saliency_3(img_sequence[2])
    up_saliency, binary_up_saliency = saliency_3(cv2.resize(img_sequence[2], None, fx = 1/resize_ratio, fy = 1/resize_ratio))
    down_saliency, binary_down_saliency = saliency_3(cv2.resize(img_sequence[3], None, fx = 1/resize_ratio, fy = 1/resize_ratio))
    #newly added
    #back_saliency, binary_back_saliency = saliency_3(cv2.resize(img_sequence[-1], None, fx = 1/resize_ratio, fy = 1/resize_ratio))
    #suppose no need to give heavy weight on the back side

    up_saliency, binary_up_saliency = cv2.resize(up_saliency, None, fx = resize_ratio, fy = resize_ratio), cv2.resize(binary_up_saliency, None, fx = resize_ratio, fy = resize_ratio)
    down_saliency, binary_down_saliency = cv2.resize(down_saliency, None, fx = resize_ratio, fy = resize_ratio), cv2.resize(binary_down_saliency, None, fx = resize_ratio, fy = resize_ratio)
    #newly added
    #back_saliency, binary_back_saliency = cv2.resize(back_saliency, None, fx = resize_ratio, fy = resize_ratio), cv2.resize(binary_back_saliency, None, fx = resize_ratio, fy = resize_ratio)

    #parameters
    img_order = [5, 1, 4, 0, 5] #align small frames in an order of ['back', 'left', 'front', 'right', 'back'] to make 360-front view
    up_rotation_list = [2, 1, 0, -1, -2] #rotation directions of up-frame
    down_rotation_list = [-2, -1, 0, 1, 2] #rotation directions of down-frame

    #make flattened cube frame
    flattened_cube_img = cube_img_concatenate(img_sequence, img_order, up_rotation_list, down_rotation_list)

    #let's extract the saliency map from the flattened cube frame
    #rbd, flattened_cube_saliency = saliency_3(flattened_cube_img)
    flattened_cube_saliency, binary_flattened_cube_saliency = saliency_3(cv2.resize(flattened_cube_img.copy(), None, fx = 1/resize_ratio, fy = 1/resize_ratio))
    flattened_cube_saliency, binary_flattened_cube_saliency = cv2.resize(flattened_cube_saliency, None, fx = resize_ratio, fy = resize_ratio), cv2.resize(binary_flattened_cube_saliency, None, fx = resize_ratio, fy = resize_ratio)

    flattened_front_view_saliency = flattened_cube_saliency[int(height/2):-int(height/2),:]
    flattened_front_view = flattened_cube_img[int(height/2):-int(height/2),:]

    #saliency union for up- and down-frame
    up_saliency, down_saliency = up_down_saliency_union(flattened_cube_saliency, up_saliency, down_saliency, up_rotation_list, down_rotation_list)

    #saliency_union map for front, left, back, and right
    row, col = flattened_front_view_saliency.shape
    window_size = int(row / salient_patch_size)
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!salient_patch_size_ratio!!!!!!!!!!!!!!!!!!!!!!!!!!!

    #slice the 4 most salient regions in inter-connected front-left-rigth-back frame
    max_sum_list, max_sum_coordinate_list = find_max_salient_regions(flattened_front_view_saliency, front_sub_square_num, window_size)

    #slice the most salient regions in up- and down-side

    up_sum_list, up_coordinate_list = up_down_salient_region(up_saliency, up_sub_square_num, window_size)
    down_sum_list, down_coordinate_list = up_down_salient_region(down_saliency, down_sub_square_num, window_size)

    #salient regions selection
    out_img, encoding_info = salient_region_selection(img_sequence, max_sum_list, max_sum_coordinate_list, up_sum_list, up_coordinate_list, down_sum_list, down_coordinate_list, flattened_front_view, flattened_front_view_saliency, num_selected_regions)

    output.put((out_img, encoding_info))
    #return out_img, encoding_info

def saliency_update(saliency_map, original_saliency_map, reverse_img_order):

    if len(saliency_map.shape) == 3:
        height, width, _ = saliency_map.shape
    else:
        height, width = saliency_map.shape

    for idx, img_order in enumerate(reverse_img_order):
        original_saliency_map[:, img_order*height:(img_order+1)*height] = np.maximum(saliency_map[:, idx*height:(idx+1)*height], original_saliency_map[:, img_order*height:(img_order+1)*height])

    return original_saliency_map
