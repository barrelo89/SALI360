import os
import cv2
import numpy as np
import multiprocessing

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

def trapezoid_1(img): # starting from small frame / 2 + 2

    if len(img.shape) == 3:
        row, col, channel = img.shape

    else:
        row, col = img.shape
        channel = 1

    empty_pallete = np.zeros((row, col, channel), dtype = np.uint8)

    for idx in range(int(row / 4)):
        layer = cv2.resize(img[idx*4:(idx+1)*4, :], (int(row/2) + 2 + 2*idx,1))
        _, length, _ = layer.shape

        empty_pallete[idx, int(row/4)-1-idx: int(row/4)-1-idx + length, :  ] = layer

        #cv2.imshow('original', img[idx*4:(idx+1)*4, :])
        #cv2.imshow('resized', layer)

    #cv2.imshow('pallete', empty_pallete)
    #cv2.waitKey(0)

    return empty_pallete #shape of row, col, channel (1024, 1024, 3)

def trapezoid_2(img): # starting from small frame / 2

    if len(img.shape) == 3:
        row, col, channel = img.shape

    else:
        row, col = img.shape
        channel = 1

    empty_pallete = np.zeros((row, col, channel), dtype = np.uint8)

    for idx in range(int(row / 4)):
        layer = cv2.resize(img[idx*4:(idx+1)*4, :], (int(row/2) + 2*idx,1))
        _, length, _ = layer.shape

        empty_pallete[idx, int(row/4)-idx: int(row/4)-idx + length, :  ] = layer

    return empty_pallete #shape of row, col, channel (1024, 1024, 3)

def pyramid_b_encoding(img, output):

    #[right_frame, left_frame, up_frame, down_frame, front_frame, back_frame]
    split_img = frame_split(img)

    row, col, channel = split_img[0].shape
    pallete = np.zeros((row, col, channel), dtype = np.uint8)

    #right: rotate left once and trapezoid_1 and rotate right, indexing from the right
    #left: rotate right once and trapezoid_1 and rotate left, indexing from the left
    #up: rotate none and trapezoid_2 rotate twice, indexing from the bottom
    #down: rotate twice and trapezoid_2, indexing from the top

    right_img = split_img[0]
    right_img = rotate_img(right_img, iteration = 1)

    right_trapezoid = trapezoid_1(right_img)
    right_trapezoid = rotate_img(right_trapezoid, iteration = -1)

    left_img = split_img[1]
    left_img = rotate_img(left_img, iteration = -1)

    left_trapezoid = trapezoid_1(left_img)
    left_trapezoid = rotate_img(left_trapezoid, iteration = 1)

    up_img = split_img[2]

    up_trapezoid = trapezoid_2(up_img)
    up_trapezoid = rotate_img(up_trapezoid, iteration = 2)


    down_img = split_img[3]

    down_img = rotate_img(down_img, iteration = 2)
    down_trapezoid = trapezoid_2(down_img)

    front_img = split_img[4]

    back_img = split_img[5]
    back_img = cv2.resize(back_img, None, fx = 0.5, fy = 0.5)

    height, width, _ = back_img.shape

    pallete[:, :int(col/4), :] = np.maximum(pallete[:, :int(col/4), :], right_trapezoid[:, -int(col/4):, :]) #right
    pallete[:, -int(col/4):, :] = np.maximum(pallete[:, -int(col/4):, :], left_trapezoid[:, :int(col/4), :]) #left
    pallete[:int(row/4), :, :] = np.maximum(pallete[:int(row/4), :, :], up_trapezoid[-int(row/4):, :, :]) #up
    pallete[-int(row/4):, :, :] = np.maximum(pallete[-int(row/4):, :, :], down_trapezoid[:int(row/4), :, :]) #down
    pallete[int(row/4):int(row/4)+height, int(col/4):int(col/4)+width, :] = back_img

    pyra_c_frame = np.concatenate([front_img, pallete], axis = 1)

    output.put(pyra_c_frame)

cube_base_path = 'video/segments/cube'
pyra_base_path = 'video/segments/pyra_b'

fourcc = cv2.VideoWriter_fourcc(*'X264')

for y_p_combo in os.listdir(cube_base_path):

    y_p_combo_path = os.path.join(cube_base_path, y_p_combo)
    y_p_write_path = os.path.join(pyra_base_path, y_p_combo)

    for duration in ['2', '5', '10']:

        duration_path = os.path.join(y_p_combo_path, duration)
        duration_write_path = os.path.join(y_p_write_path, duration)

        if not os.path.exists(duration_write_path):
            os.makedirs(duration_write_path)

        for file_name in os.listdir(duration_path):

            file_path = os.path.join(duration_path, file_name)
            file_write_path = os.path.join(duration_write_path, file_name)

            capture = cv2.VideoCapture(file_path)

            fps = round(capture.get(cv2.CAP_PROP_FPS))
            frame_input_list = []
            frame_idx_list = []

            outputs = []
            process = []

            num_core_2_use = 15

            frame_idx = 0

            while(capture.isOpened()):

                retval, frame = capture.read()

                if retval == True:

                    frame_input_list.append(frame)
                    frame_idx_list.append(frame_idx)

                    if len(frame_input_list) == num_core_2_use:

                        for _ in range(num_core_2_use):
                            outputs.append(multiprocessing.Queue())

                        for order in range(num_core_2_use):
                            process.append(multiprocessing.Process(target = pyramid_b_encoding, args = (frame_input_list[order], outputs[order])))

                        for idx, pro in enumerate(process):
                            pro.start()

                        for idx in range(num_core_2_use):
                            pyra_b_frame = outputs[idx].get()

                            if frame_idx_list[idx] == 0:
                                frame_height, frame_width, _ = pyra_b_frame.shape
                                writer = cv2.VideoWriter(file_write_path, fourcc, fps, (frame_width, frame_height))

                            writer.write(pyra_b_frame)

                        for idx in range(num_core_2_use):
                            outputs[idx].close()

                        for idx, pro in enumerate(process):
                            pro.terminate()

                        outputs = []
                        process = []
                        frame_input_list = []
                        frame_idx_list = []

                    frame_idx += 1
                else:
                    break

            capture.release()
            writer.release()

















































#end
