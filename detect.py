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
threshhold = None
fingerDetector = None

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
    global detection_graph, sess, fingerDetector
    detection_graph, sess = detector_utils.load_inference_graph()
    fingerDetector = ModelPipeline()


def detectHands(frame):
    global detection_graph, sess, lhands, threshhold
    boxes, scores = detector_utils.detect_objects(
        cv2.cvtColor(cv2.resize(frame, (90, 60)), cv2.COLOR_BGR2RGB), detection_graph, sess)

    hands = [None, None]

    if scores[0] > 0.3:
        hands[0] = boxes[0]
        hands[0][0] = int(hands[0][0] * 480)
        hands[0][1] = int(hands[0][1] * 640)
        hands[0][2] = int(hands[0][2] * 480)
        hands[0][3] = int(hands[0][3] * 640)
        hands[0] = hands[0].astype(np.int)

        if scores[1] > 0.3:
            hands[1] = boxes[1]
            hands[1][0] = int(hands[1][0] * 480)
            hands[1][1] = int(hands[1][1] * 640)
            hands[1][2] = int(hands[1][2] * 480)
            hands[1][3] = int(hands[1][3] * 640)
            hands[1] = hands[1].astype(np.int)
    return hands


def detectFinger(framebox):
    global fingerDetector
    isLeft = True
    cords = fingerDetector.process(framebox)[0]
    cords = np.delete(cords, 2, 1)
    cords = cords * 50 + 60
    return (cords, isLeft)


def handboxToFramebox(frame, hand):
    detectbox = np.array([[hand[0], hand[2]], [hand[1], hand[3]]]).clip(min=1)
    detectbox[0] = detectbox[0].clip(max=480)
    detectbox[1] = detectbox[1].clip(max=640)
    detection = frame[detectbox[0][0]:detectbox[0][1], detectbox[1][0]:detectbox[1][1]]
    detection = cv2.resize(detection, (128, 128))
    return detection


__init__()

cam = cv2.VideoCapture('http://192.168.178.195:4747/mjpegfeed?640x480')

while True:
    img = cam.read()[1]
    boxes = detectHands(img)

    finger = [None, None]

    if boxes[0] is not None:
        finger[0] = detectFinger(handboxToFramebox(img, boxes[0])) 
    if boxes[1] is not None:
        finger[1] = detectFinger(handboxToFramebox(img, boxes[1]))

    if boxes[0] is not None:
        cv2.rectangle(img, (boxes[0][1], boxes[0][0]),
                      (boxes[0][3], boxes[0][2]), (77, 255, 9), 3, 1)
        cv2.putText(img, "HAND 1", (boxes[0][1], boxes[0][0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255))
        for x in finger[0][0]:
            cv2.drawMarker(img, (int(x[0]+boxes[0][1]), int(x[1]+boxes[0][0])), (0, 0, 0))
    if boxes[1] is not None:
        cv2.rectangle(img, (boxes[1][1], boxes[1][0]),
                      (boxes[1][3], boxes[1][2]), (77, 255, 9), 3, 1)
        cv2.putText(img, "HAND 2", (boxes[1][1], boxes[1][0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255))
        for x in finger[1][0]:
            cv2.drawMarker(img, (int(x[0]+boxes[1][1]), int(x[1]+boxes[1][0])), (0, 0, 0))

    cv2.imshow("Output", img)
    cv2.waitKey(1)
    time.sleep(0.015)
