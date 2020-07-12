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


def Loop1(q, q1, q2):
    global lboxes, detection_graph, sess, clf, num_frames, num_hands_detect, smoother1, smoother2, cords_smoother1, cords_smoother2, mesh_smoother1, mesh_smoother2, imgi, dery, mesh, viewer1, viewer2, model

    while True:
        if q.empty():
            continue

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        ret, image_np = q.get()
        # image_np = cv2.flip(image_np, 1)
        try:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        except:
            print("Error converting to RGB")

        # Actual detection. Variable boxes contains the bounding box cordinates for hands detected,
        # while scores contains the confidence for each of these boxes.
        # Hint: If len(boxes) > 1 , you may assume you have found atleast one hand (within your score threshold)

        boxes, scores = detector_utils.detect_objects(
            image_np, detection_graph, sess)

        if lboxes is None:
            lboxes = boxes

        tmp = lboxes
        lboxes = []
        accuracy = []
        back = [False, False]

        # if tmp:
        # tmp[0] = smoother1.process(tmp[0])
        # tmp[1] = smoother2.process(tmp[1])

        if len(tmp) >= 2:
            if scores[0] > 0.2 and scores[1] > 0.2:
                if np.mean(tmp[1] != boxes[0]) < np.mean(tmp[0] != boxes[1]):
                    print("change")
                    tmp2 = scores[1]
                    scores[1] = scores[0]
                    scores[0] = tmp2
                    tmp2 = boxes[1]
                    boxes[1] = boxes[0]
                    boxes[0] = tmp2

        for x in range(2):
            if scores[x] > 0.0:
                res = [(boxes[x][2] * 480 - boxes[x][0] * 480),
                       (boxes[x][3] * 640 - boxes[x][1] * 640)]
                if res[0] > res[1]:
                    res = (res[0]+50 - res[1])/2/480
                    boxes[x][1] -= res
                    boxes[x][3] += res
                elif res[0] < res[1]:
                    res = (res[1] - res[0])/2/640
                    boxes[x][0] -= res
                    boxes[x][2] += res

                if len(tmp) > x and len(boxes) > x:
                    accuracy.append(np.mean(tmp[x] != boxes[x]))
                lboxes.append(boxes[x])

        res = [True, True]

        if (boxes[0][1] > boxes[1][1] and boxes[0][1] < boxes[1][3] and boxes[0][0] > boxes[1][0] and boxes[0][0] < boxes[1][2]):
            if (boxes[0][3] > boxes[1][1] and boxes[0][3] < boxes[1][3] and boxes[0][2] > boxes[1][0] and boxes[0][2] < boxes[1][2]):
                res[0] = False
        if (boxes[1][1] > boxes[0][1] and boxes[1][1] < boxes[0][3] and boxes[1][0] > boxes[0][0] and boxes[1][0] < boxes[0][2]):
            if (boxes[1][3] > boxes[0][1] and boxes[1][3] < boxes[0][3] and boxes[1][2] > boxes[0][0] and boxes[1][2] < boxes[0][2]):
                res[1] = False

        if res[0] and res[1]:
            res[0] = False
            res[1] = False

        i = 0

        for x in range(len(accuracy)):
            if res[x] is True:
                if accuracy[x+i] < 0.4:
                    accuracy.pop(x+i)
                    i -= 1
                else:
                    if back[x+i] is True:
                        back[x+i] = False
                        accuracy.pop(x+i)
                        i -= 1
                    else:
                        back[x+i] = True

            # if accuracy:
            #     if back[0] is True or back[1] is True:
            #         if int(i/2) == i/2:
            #             for x in range(len(tmp)):
            #                 if back[x] is True:
            #                     with open('tdata/labels.txt', 'a') as file:
            #                         file.write(f"{str(imgi/2).zfill(6)} {x} {lboxes[x][0]} {lboxes[x][1]} {lboxes[x][2]} {lboxes[x][3]}\n")

            #             predimg = image_np.copy()
            #             labelimg = image_np.copy()

            #             detector_utils.draw_box_on_image(num_hands_detect, args.score_thresh,
            #                                     scores, tmp, 640, 480,
            #                                     predimg)

            #             detector_utils.draw_box_on_image(num_hands_detect, args.score_thresh,
            #                                     scores, lboxes, 640, 480,
            #                                     labelimg)

            #             cv2.imwrite(f"tdata/{str(imgi/2).zfill(6)}.png", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
            #             cv2.imwrite(f"tdata/utils/predicted/{str(imgi/2).zfill(6)}.png", cv2.cvtColor(predimg, cv2.COLOR_RGB2BGR))
            #             cv2.imwrite(f"tdata/utils/label/{str(imgi/2).zfill(6)}.png", cv2.cvtColor(labelimg, cv2.COLOR_RGB2BGR))

            #         imgi += 1

        if len(boxes) > 0:
            detectbox = np.array([[[int(boxes[0][0] * 480), int(boxes[0][2] * 480)], [int(boxes[0][1] * 640), int(boxes[0][3] * 640)]],
                                  [[int(boxes[1][0] * 480), int(boxes[1][2] * 480)], [int(boxes[1][1] * 640), int(boxes[1][3] * 640)]]]).clip(min=1)
            detectbox[0][0] = detectbox[0][0].clip(max=480)
            detectbox[1][0] = detectbox[1][0].clip(max=480)
            detectbox[0][1] = detectbox[0][1].clip(max=640)
            detectbox[1][1] = detectbox[1][1].clip(max=640)
            detection = [image_np.copy()[detectbox[0][0][0]:detectbox[0][0][1], detectbox[0][1][0]:detectbox[0][1][1]],
                         image_np.copy()[detectbox[1][0][0]:detectbox[1][0][1], detectbox[1][1][0]:detectbox[1][1][1]]]
            try:
                detection[0] = cv2.resize(detection[0], (128, 128))
                dery[0] = True
            except:
                dery[0] = False
            try:
                detection[1] = cv2.resize(detection[1], (128, 128))
                dery[1] = True
            except:
                dery[1] = False

            if dery[0] is True:
                if scores[0] < 0.2:
                    dery[0] = False

            if dery[1] is True:
                if scores[1] < 0.2:
                    dery[1] = False
        # draw bounding boxes on frame
        detector_utils.draw_box_on_image(num_hands_detect, 0.2,
                                         scores, boxes, 640, 480,
                                         image_np)

        # Calculate Frames per second (FPS)
        num_frames += 1
        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        fps = num_frames / elapsed_time

        if dery[0] is True:
            if q1.qsize() <= 2:
                q1.put(detection[0].copy())
        if dery[1] is True:
            if q2.qsize() <= 2:
                q2.put(detection[1].copy())

        cv2.imshow('Single-Threaded Detection',
                   cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

        time.sleep(0.0016)


def Loop2(q):
    global lboxes, detection_graph, sess, clf, num_frames, num_hands_detect, smoother1, smoother2, cords_smoother1, cords_smoother2, mesh_smoother1, mesh_smoother2, imgi, mesh, viewer1, viewer2, model

    while True:
        if q.empty():
            continue

        # cv2.imshow('Detection', cv2.cvtColor(detection, cv2.COLOR_RGB2BGR))

        frame_large = q.get()
        # if frame_large is None:
        # continue
        # if frame_large.shape[0] > frame_large.shape[1]:
        # margin = int((frame_large.shape[0] - frame_large.shape[1]) / 2)
        # frame_large = frame_large[margin:-margin]
        # else:
        # margin = int((frame_large.shape[1] - frame_large.shape[0]) / 2)
        # frame_large = frame_large[:, margin:-margin]

        frame_large = np.flip(frame_large, axis=1).copy()
        frame = frame_large.copy()

        cords, theta_mpii = model.process(frame)
        # theta_mano = mpii_to_mano(theta_mpii)

        # v = hand_mesh.set_abs_quat(theta_mano)
        # v *= 2 # for better visualization
        # v = v * 1000 + np.array([0, 0, 400])
        # v = np.array([x+cords[0] for x in v])
        # v = mesh_smoother1.process(v)
        # mesh.triangles = o3d.utility.Vector3iVector(hand_mesh.faces)
        # mesh.vertices = o3d.utility.Vector3dVector(np.matmul(view_mat, v.T).T)
        # mesh.paint_uniform_color(config.HAND_COLOR)
        # mesh.compute_triangle_normals()
        # mesh.compute_vertex_normals()
        # for some version of open3d you may need `viewer.update_geometry(mesh)`
        # viewer1.update_geometry()

        # viewer1.poll_events()

        # cords = cords_smoother1.process(cords)

        cords = cords * 50 + 60
        cords = np.delete(cords, 2, 1)

        # v = v * 1.5 + 200
        # v = np.delete(v, 2, 1)

        frame_large = cv2.cvtColor(frame_large, cv2.COLOR_RGB2BGR)
        # cv2.polylines(frame_large, v, False, (0, 0, 0))
        for x in cords:
            cv2.drawMarker(frame_large, (int(x[0]), int(x[1])), (0, 0, 0))
            # cv2.line(frame_large, (int(v[x][0]), int(v[x][1])), (int(v[x+1][0]), int(v[x+1][1])), (0, 0, 0))

        # meshindices = np.array(mesh.triangles)
        # meshvertices = np.array(mesh.vertices) - 80

        # pts2d = cv2.projectPoints(meshvertices, (0, 0, 0), (0, 0, 0), np.array([[620.744, 0., 0.], [0., 621.151, 0.], [0., 0., 1.]]), None)[0].astype(int)
        # for face in meshindices:
        #     cv2.fillConvexPoly(frame_large, pts2d[face], (64, 64, 192))
        # cv2.polylines(frame_large, pts2d[meshindices], True, (255, 255, 255))

        # cv2.polylines(frame_large, v, False, (0, 0, 0))
        cv2.imshow("Hand AI Left", cv2.resize(frame_large, (480, 480)))

        cv2.waitKey(1)

        time.sleep(0.0016)


def Loop3(q):
    global lboxes, detection_graph, sess, clf, num_frames, num_hands_detect, smoother1, smoother2, cords_smoother1, cords_smoother2, mesh_smoother1, mesh_smoother2, imgi, mesh, viewer1, viewer2, model

    while True:
        if q.empty():
            continue

        # cv2.imshow('Detection', cv2.cvtColor(detection, cv2.COLOR_RGB2BGR))

        frame_large = q.get()
        # if frame_large is None:
        # continue
        # if frame_large.shape[0] > frame_large.shape[1]:
        # margin = int((frame_large.shape[0] - frame_large.shape[1]) / 2)
        # frame_large = frame_large[margin:-margin]
        # else:
        # margin = int((frame_large.shape[1] - frame_large.shape[0]) / 2)
        # frame_large = frame_large[:, margin:-margin]

        # frame_large = np.flip(frame_large, axis=1).copy()
        frame = frame_large.copy()

        cords, theta_mpii = model.process(frame)
        # theta_mano = mpii_to_mano(theta_mpii)

        # v = hand_mesh.set_abs_quat(theta_mano)
        # v *= 2 # for better visualization
        # v = v * 1000 + np.array([0, 0, 400])
        # v = np.array([x+cords[0] for x in v])
        # v = mesh_smoother2.process(v)
        # mesh.triangles = o3d.utility.Vector3iVector(hand_mesh.faces)
        # mesh.vertices = o3d.utility.Vector3dVector(np.matmul(view_mat, v.T).T)
        # mesh.paint_uniform_color(config.HAND_COLOR)
        # mesh.compute_triangle_normals()
        # mesh.compute_vertex_normals()
        # for some version of open3d you may need `viewer.update_geometry(mesh)`
        # viewer2.update_geometry()

        # viewer2.poll_events()

        # cords = cords_smoother2.process(cords)

        cords = cords * 50 + 60
        cords = np.delete(cords, 2, 1)

        # v = v * 1.5 + 200
        # v = np.delete(v, 2, 1)

        frame_large = cv2.cvtColor(frame_large, cv2.COLOR_RGB2BGR)
        # cv2.polylines(frame_large, v, False, (0, 0, 0))
        for x in cords:
            cv2.drawMarker(frame_large, (int(x[0]), int(x[1])), (0, 0, 0))
            # cv2.line(frame_large, (int(v[x][0]), int(v[x][1])), (int(v[x+1][0]), int(v[x+1][1])), (0, 0, 0))

        # meshindices = np.array(mesh.triangles)
        # meshvertices = np.array(mesh.vertices) - 80

        # pts2d = cv2.projectPoints(meshvertices, (0, 0, 0), (0, 0, 0), np.array([[620.744, 0., 0.], [0., 621.151, 0.], [0., 0., 1.]]), None)[0].astype(int)
        # for face in meshindices:
        #     cv2.fillConvexPoly(frame_large, pts2d[face], (64, 64, 192))
        # cv2.polylines(frame_large, pts2d[meshindices], True, (255, 255, 255))

        # cv2.polylines(frame_large, v, False, (0, 0, 0))
        cv2.imshow("Hand AI Right", cv2.resize(frame_large, (480, 480)))

        cv2.waitKey(1)

        time.sleep(0.0016)


detection_graph, sess = detector_utils.load_inference_graph()

clf = svm.SVC(random_state=0)
cap = cv2.VideoCapture(1)
ret, image_np = cap.read()

lboxes = []

start_time = datetime.datetime.now()
num_frames = 0
# max number of hands we want to detect/track
num_hands_detect = 2

smoother1 = OneEuroFilter(4.0, 0.0)
smoother2 = OneEuroFilter(4.0, 0.0)

# cv2.namedWindow('Single-Threaded Detection', cv2.WINDOW_NORMAL)

imgi = 0
dery = [False, False]
lboxes = None

view_mat = axangle2mat([1, 0, 0], np.pi)
window_size_w = 640
window_size_h = 480

hand_mesh = HandMesh(config.HAND_MESH_MODEL_PATH)
mesh = o3d.geometry.TriangleMesh()
mesh.triangles = o3d.utility.Vector3iVector(hand_mesh.faces)
mesh.vertices = \
    o3d.utility.Vector3dVector(np.matmul(view_mat, hand_mesh.verts.T).T * 1000)
mesh.compute_vertex_normals()

# viewer1 = o3d.visualization.Visualizer()
# viewer1.create_window(
#     width=window_size_w + 1, height=window_size_h + 1,
#     window_name='Minimal Hand - output'
# )
# viewer1.add_geometry(mesh)

# viewer2 = o3d.visualization.Visualizer()
# viewer2.create_window(
#     width=window_size_w + 1, height=window_size_h + 1,
#     window_name='Minimal Hand - output'
# )
# viewer2.add_geometry(mesh)

# view_control = viewer1.get_view_control()
# cam_params = view_control.convert_to_pinhole_camera_parameters()
# extrinsic = cam_params.extrinsic.copy()
# extrinsic[0:3, 3] = 0
# cam_params.extrinsic = extrinsic
# cam_params.intrinsic.set_intrinsics(
#     window_size_w + 1, window_size_h + 1, config.CAM_FX, config.CAM_FY,
#     window_size_w // 2, window_size_h // 2
# )
# view_control.convert_from_pinhole_camera_parameters(cam_params)
# view_control.set_constant_z_far(1000)

# view_control = viewer2.get_view_control()
# cam_params = view_control.convert_to_pinhole_camera_parameters()
# extrinsic = cam_params.extrinsic.copy()
# extrinsic[0:3, 3] = 0
# cam_params.extrinsic = extrinsic
# cam_params.intrinsic.set_intrinsics(
#     window_size_w + 1, window_size_h + 1, config.CAM_FX, config.CAM_FY,
#     window_size_w // 2, window_size_h // 2
# )
# view_control.convert_from_pinhole_camera_parameters(cam_params)
# view_control.set_constant_z_far(1000)

# render_option = viewer1.get_render_option()
# render_option.load_from_json('./render_option.json')
# viewer1.update_renderer()
# render_option = viewer2.get_render_option()
# render_option.load_from_json('./render_option.json')
# viewer2.update_renderer()

mesh_smoother1 = OneEuroFilter(2.0, 0.0)
mesh_smoother2 = OneEuroFilter(2.0, 0.0)
cords_smoother1 = OneEuroFilter(2.0, 0.0)
cords_smoother2 = OneEuroFilter(2.0, 0.0)
model = ModelPipeline()

que = Queue()
quet1 = Queue()
quet2 = Queue()


t11 = threading.Thread(target=Loop1, args=(que, quet1, quet2, ))
t12 = threading.Thread(target=Loop1, args=(que, quet1, quet2, ))
t21 = threading.Thread(target=Loop2, args=(quet1, ))
t22 = threading.Thread(target=Loop2, args=(quet1, ))
t31 = threading.Thread(target=Loop3, args=(quet2, ))
t32 = threading.Thread(target=Loop3, args=(quet2, ))
# t4 = threading.Thread(target=mainLoop, args=(que, ))
t11.start()
t12.start()
t21.start()
t22.start()
t31.start()
t32.start()
# t4.start()

while True:
    if que.qsize() <= 2:
        que.put(cap.read())
        time.sleep(0.0016)
    else:
        time.sleep(0.0016)
