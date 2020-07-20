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
        frame, detection_graph, sess)

    hands = [None, None]

    if scores[0] > 0.02:
        hands[0] = boxes[0]
        hands[0][0] = int(hands[0][0] * 640)
        hands[0][1] = int(hands[0][1] * 480)
        hands[0][2] = int(hands[0][2] * 640)
        hands[0][3] = int(hands[0][3] * 480)

        if scores[1] > 0.2:
            hands[1] = boxes[1]
            hands[1][0] = int(hands[1][0] * 640)
            hands[1][1] = int(hands[1][1] * 480)
            hands[1][2] = int(hands[1][2] * 640)
            hands[1][3] = int(hands[1][3] * 480)

    
    isNone = [hands[0] is not None, hands[1] is not None]
    lisNone = [lhands[0] is not None, lhands[1] is not None]

    if isNone[0] and isNone[1]:
        acc = [None, None]
        if lisNone[0]:
            acc[0] = np.mean(hands[0] == lhands[0])
        if lisNone[1]:
            acc[1] = np.mean(hands[0] == lhands[1])
        
        if lisNone[0] and lisNone[1]:
            if acc[0] < acc[1]:
                hands = hands[::-1]
    elif isNone[0] or isNone[1]:
        acc = [None, None]
        hand = None

        if isNone[0]:
            hand = 0
        elif isNone[1]:
            hand = 1
        else:
            raise Exception("ArgumentError while Loop")

        if lisNone[0]:
            acc[0] = np.mean(hands[hand] == lhands[0])
        if lisNone[1]:
            acc[1] = np.mean(hands[hand] == lhands[1])
        
        if lisNone[0] and lisNone[1]:
            if acc[0] > acc[1]:
                if hand == 1:
                    hands = hands[::-1]
            else:
                if hand == 0:
                    hands = hands[::-1]
        # else:
        #     if lisNone[hand]:
        #         if acc[hand] < 0.6:
        #             hands = hands.reverse()

    lhands = hands

    return hands


__init__()

cam = cv2.VideoCapture('http://192.168.178.195:4747/mjpegfeed?640x480')

while True:
    img = cam.read()[1]
    boxes = detectHands(img)

    if boxes[0] is not None:
        cv2.rectangle(img, (boxes[0][1], boxes[0][0]), (boxes[0][3], boxes[0][2]), (0, 0, 255))
    if boxes[1] is not None:
        cv2.rectangle(img, (boxes[1][1], boxes[1][0]), (boxes[1][3], boxes[1][2]), (0, 0, 255))
    
    cv2.imshow("Output", img)
    cv2.waitKey(1)