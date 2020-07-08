import cv2
import keyboard
import numpy as np
import open3d as o3d
import time
from transforms3d.axangles import axangle2mat

import config
from capture import OpenCVCapture
from hand_mesh import HandMesh
from kinematics import mpii_to_mano
from util import OneEuroFilter, imresize
from wrappers import ModelPipeline
from util import *


def live_application(capture):
    """
    Launch an application that reads from a webcam and estimates hand pose at
    real-time.

    The captured hand must be the right hand, but will be flipped internally
    and rendered.

    Parameters
    ----------
    capture : object
        An object from `capture.py` to read capture stream from.
    """

    view_mat = axangle2mat([1, 0, 0], np.pi)
    window_size_w = 128
    window_size_h = 128

    hand_mesh = HandMesh(config.HAND_MESH_MODEL_PATH) 
    mesh = o3d.geometry.TriangleMesh()
    mesh.triangles = o3d.utility.Vector3iVector(hand_mesh.faces)
    mesh.vertices = \
        o3d.utility.Vector3dVector(np.matmul(view_mat, hand_mesh.verts.T).T * 1000)
    mesh.compute_vertex_normals()

    viewer = o3d.visualization.Visualizer()
    viewer.create_window(
        width=window_size_w + 1, height=window_size_h + 1,
        window_name='Minimal Hand - output'
    )
    viewer.add_geometry(mesh)

    view_control = viewer.get_view_control()
    cam_params = view_control.convert_to_pinhole_camera_parameters()
    extrinsic = cam_params.extrinsic.copy()
    extrinsic[0:3, 3] = 0
    cam_params.extrinsic = extrinsic
    cam_params.intrinsic.set_intrinsics(
        window_size_w + 1, window_size_h + 1, config.CAM_FX, config.CAM_FY,
        window_size_w // 2, window_size_h // 2
    )
    view_control.convert_from_pinhole_camera_parameters(cam_params)
    view_control.set_constant_z_far(1000)

    render_option = viewer.get_render_option()
    render_option.load_from_json('./render_option.json')
    viewer.update_renderer()

    mesh_smoother = OneEuroFilter(4.0, 0.0)
    cords_smoother = OneEuroFilter(4.0, 0.0)
    model = ModelPipeline()

    while True:
        frame_large = capture.read()
        if frame_large is None:
            continue
        if frame_large.shape[0] > frame_large.shape[1]:
            margin = int((frame_large.shape[0] - frame_large.shape[1]) / 2)
            frame_large = frame_large[margin:-margin]
        else:
            margin = int((frame_large.shape[1] - frame_large.shape[0]) / 2)
            frame_large = frame_large[:, margin:-margin]

        if config.left is False:
            frame_large = np.flip(frame_large, axis=1).copy()
        frame = imresize(frame_large, (128, 128))

        cords, theta_mpii = model.process(frame)
        theta_mano = mpii_to_mano(theta_mpii)

        v = hand_mesh.set_abs_quat(theta_mano)
        # v *= 2 # for better visualization
        v = v * 1000 + np.array([0, 0, 400])
        # v = np.array([x+cords[0] for x in v])
        v = mesh_smoother.process(v)
        mesh.triangles = o3d.utility.Vector3iVector(hand_mesh.faces)
        mesh.vertices = o3d.utility.Vector3dVector(np.matmul(view_mat, v.T).T)
        mesh.paint_uniform_color(config.HAND_COLOR)
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()
        # for some version of open3d you may need `viewer.update_geometry(mesh)`
        viewer.update_geometry()

        viewer.poll_events()

        cords = cords_smoother.process(cords)

        cords = cords * 150 + 200
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
        cv2.imshow("Hand AI", frame_large)

        if keyboard.is_pressed("q"):
            cv2.destroyAllWindows()
            break

        cv2.waitKey(1)
        time.sleep(0.0015)


if __name__ == '__main__':
    live_application(OpenCVCapture())
