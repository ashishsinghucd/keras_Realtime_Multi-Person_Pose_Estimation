import os
import sys
import argparse
import cv2
import time

import pandas as pd

from config_reader import config_reader
from processing import extract_parts
from model.cmu_model import get_testing_model
from util import create_directory_if_not_exists


def load_model(keras_weights_file):
    model = get_testing_model()
    model.load_weights(keras_weights_file)
    params, model_params = config_reader()
    return model, params, model_params


def generate_cooridnates(extracted_coordinates_path, clip_file_path, keras_weights_file, ending_frame=None):
    video_file_name = clip_file_path.rsplit("/", 1)[1].rsplit(".", 1)[0]
    if os.path.isfile(extracted_coordinates_path + "/" + video_file_name + "_x.csv"):
        return

    model, params, model_params = load_model(keras_weights_file)

    scale_search = [1]
    params['scale_search'] = scale_search

    cam = cv2.VideoCapture(clip_file_path)
    input_fps = cam.get(cv2.CAP_PROP_FPS)

    video_length = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total frame in the video: ", video_length)
    if ending_frame is None:
        ending_frame = video_length

    ret_val, orig_image = cam.read()
    i = 0  # default is 0
    cc = []
    start = time.time()
    while (cam.isOpened()) and ret_val is True and i < ending_frame:
        input_image = cv2.cvtColor(orig_image, cv2.COLOR_RGB2BGR)
        tic = time.time()
        # generate image with body parts
        body_parts, all_peaks, subset, candidate = extract_parts(input_image, params, model, model_params)
        # print('Processing frame: ', i)
        toc = time.time()
        # print('Processing time is %.5f' % (toc - tic))
        cc.append(body_parts)
        ret_val, orig_image = cam.read()
        i += 1

    xx = [{k: v[0] for k, v in d.items()} for d in cc]
    yy = [{k: v[1] for k, v in d.items()} for d in cc]

    xx_ = pd.DataFrame(xx)
    yy_ = pd.DataFrame(yy)

    end = time.time()
    print('Total time taken is %.5f' % (end - start))
    xx_.to_csv(extracted_coordinates_path + "/" + video_file_name + "_x.csv", index=False)
    yy_.to_csv(extracted_coordinates_path + "/" + video_file_name + "_y.csv", index=False)


def generate_coordinates_helper(extracted_coordinates_path, extracted_clips_path, keras_weights_file):
    for root, dirs, files in os.walk(extracted_clips_path):
        for name in files:
            try:
                print("Extracting the coordinates of: {}".format(name))
                clip_file_path = os.path.join(root, name)
                generate_cooridnates(extracted_coordinates_path, clip_file_path, keras_weights_file)
            except Exception as e:
                print("Error extracting the coordinates of:; {} {}".format(name, str(e)))


if __name__ == "__main__":
    base_path = sys.argv[1]
    exercise = sys.argv[2]
    coordinates_path = "FullCoordinates"
    clips_path = "ExtractedClips"
    extracted_coordinates_path = os.path.join(base_path, coordinates_path, exercise)
    create_directory_if_not_exists(extracted_coordinates_path)
    extracted_clips_path = os.path.join(base_path, clips_path, exercise)
    keras_weights_file = "model.h5"
    generate_coordinates_helper(extracted_coordinates_path, extracted_clips_path, keras_weights_file)
