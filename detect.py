from utils import detector_utils as detector_utils
from sklearn import svm
import cv2
import math
import time
import numpy as np
import tensorflow as tf
import datetime
import argparse
import keyboard
import open3d as o3d
import time
from transforms3d.axangles import axangle2mat

import config
import threading
from multiprocessing import Queue
from hand_mesh import HandMesh
from kinematics import mpii_to_mano
from util import imresize
from wrappers import ModelPipeline
from utils import *

detection_graph = None
sess = None

lhands = [None, None]

class LowPassFilter:
    def __init__(self):
        self.prev_raw_value = None
        self.prev_filtered_value = None

    def process(self, value, alpha):
        if self.prev_raw_value is None:
            s = value
        else:
            s = alpha * value + (1.0 - alpha) * self.prev_filtered_value
        self.prev_raw_value = value
        self.prev_filtered_value = s
        return s


class OneEuroFilter:
    def __init__(self, mincutoff=1.0, beta=0.0, dcutoff=1.0, freq=30):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()

    def compute_alpha(self, cutoff):
        te = 1.0 / self.freq
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def process(self, x):
        prev_x = self.x_filter.prev_raw_value
        dx = 0.0 if prev_x is None else (x - prev_x) * self.freq
        edx = self.dx_filter.process(dx, self.compute_alpha(self.dcutoff))
        cutoff = self.mincutoff + self.beta * np.abs(edx)
        return self.x_filter.process(x, self.compute_alpha(cutoff))


def __init__():
    global detection_graph, sess
    detection_graph, sess = detector_utils.load_inference_graph()


def detectHands(frame):
    global detection_graph, sess, lhands
    boxes, scores = detector_utils.detect_objects(
        image_np, detection_graph, sess)

    hands = [None, None]

    for x in range(2):
        if scores[x] > 0.2:
            hands[x] = boxes[x]

    if hands[0] is not None and hands[1] is not None:
        pass
    else:
        pass
    
    lhands = hands.copy()

    return hands
