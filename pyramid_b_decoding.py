'''
MIT License

Copyright (c) [2020] [Duin BAEK]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import numpy as np
import cv2
import os

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

def trapezoid_1_decoding(img): # starting from small frame / 2 + 2

    if len(img.shape) == 3:
        row, col, channel = img.shape

    else:
        row, col = img.shape
        channel = 1

    empty_pallete = np.zeros((row, col, channel), dtype = np.uint8)

    for idx in range(int(col / 4)):

        valid_layer = img[idx:row-idx, idx, :]
        valid_row, valid_col = valid_layer.shape

        layer = cv2.resize(valid_layer.reshape(valid_row, -1, channel), (4, row))

        empty_pallete[:, idx*4:(idx+1)*4] = layer

    return empty_pallete #shape of row, col, channel (1024, 1024, 3)

def trapezoid_2_decoding(img): # starting from small frame / 2

    if len(img.shape) == 3:
        row, col, channel = img.shape

    else:
        row, col = img.shape
        channel = 1

    empty_pallete = np.zeros((row, col, channel), dtype = np.uint8)

    for idx in range(int(row / 4)):

        valid_layer = img[255-idx, idx+1:col-1-idx, :]
        valid_row, valid_col = valid_layer.shape

        layer = cv2.resize(valid_layer.reshape(-1, valid_row, channel), (col, 4))

        empty_pallete[(255-idx)*4: (255-idx+1)*4, :] = layer

    return empty_pallete #shape of row, col, channel (1024, 1024, 3)

def pyramid_b_decoding(img):

    row, col, channel = img.shape

    front_img = img[:, :int(col/2)]
    pyramid_img = img[:, int(col/2):]

    height, width, channel = pyramid_img.shape

    back_img = pyramid_img[int(height/4):int(height/4)+int(height/2), int(width/4):int(width/4)+int(width/2)]
    back_img = cv2.resize(back_img, None, fx = 2, fy = 2)

    right_img = pyramid_img[:, :int(width/4), :]
    right_pallete = np.zeros(pyramid_img.shape, dtype = np.uint8)
    right_pallete[:, :int(width/4), :] = right_img

    right_img = trapezoid_1_decoding(right_pallete)

    left_img = cv2.flip(pyramid_img[:, -int(width/4):, :], 1)
    left_pallete = np.zeros(pyramid_img.shape, dtype = np.uint8)
    left_pallete[:, :int(width/4), :] = left_img

    left_img = trapezoid_1_decoding(left_pallete)
    left_img = cv2.flip(left_img, 1)


    up_img = pyramid_img[:int(height/4), :, :]
    up_pallete = np.zeros(pyramid_img.shape, dtype = np.uint8)
    up_pallete[-int(height/4):, :, :] = up_img
    up_pallete = rotate_img(up_pallete, iteration = 2)

    up_img = trapezoid_2_decoding(up_pallete)

    down_img = pyramid_img[-int(height/4):, :, :]
    down_pallete = np.zeros(pyramid_img.shape, dtype = np.uint8)
    down_pallete[:int(height/4), :, :] = down_img

    down_img = trapezoid_2_decoding(down_pallete)
    down_img = rotate_img(down_img, 2)

    first_layer = np.concatenate([right_img, left_img, up_img], axis = 1)
    second_layer = np.concatenate([down_img, front_img, back_img], axis = 1)
    output = np.concatenate([first_layer, second_layer], axis = 0)

    return output

video_path = 'video/segments/pyra'
decoded_path = 'video/segments/pyra_decoded'

fourcc = cv2.VideoWriter_fourcc(*'X264')

for y_p_combo in os.listdir(video_path):

    y_p_combo_path = os.path.join(video_path, y_p_combo)
    y_p_write_path = os.path.join(decoded_path, y_p_combo)

    for duration in os.listdir(y_p_combo_path):

        duration_path = os.path.join(y_p_combo_path, duration)
        duration_write_path = os.path.join(y_p_write_path, duration)

        if not os.path.exists(duration_write_path):
            os.makedirs(duration_write_path)

        for file_name in os.listdir(duration_path):

            file_path = os.path.join(duration_path, file_name)
            file_write_path = os.path.join(duration_write_path, file_name)

            capture = cv2.VideoCapture(file_path)
            fps = round(capture.get(cv2.CAP_PROP_FPS))
            frame_idx = 0

            while(capture.isOpened()):

                retval, frame = capture.read()
                if retval == True:

                    pyramid_b_cube = pyramid_b_decoding(frame)

                    if frame_idx == 0:
                        frame_height, frame_width, _ = pyramid_b_cube.shape
                        writer = cv2.VideoWriter(file_write_path, fourcc, fps, (frame_width, frame_height))

                    writer.write(pyramid_b_cube)
                    frame_idx += 1
                else:
                    break

            capture.release()
            writer.release()
























#end
