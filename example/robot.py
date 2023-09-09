import sys
from cortano import RemoteInterface
import pygame
from pygame.locals import *
import time
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision import transforms
from PIL import Image
import torch
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from pyapriltags import Detector
import open3d

def bringArmToHighPosition():

    print("\n\n============= entering bringArmToHighPosition() ===================\n\n")

    cumErr = 0

    while True:      
        robot.update() # must never be forgotten
        _, _, sensors = robot.read()

        if len(sensors) == 4:
            _, armPosition, _, _ = sensors

            print(f"current armPostion: {armPosition}")

            # move arm to top position
            if armPosition >= armHeightAtHighPosition * 0.9:
                robot.motor[8] = 0
                break
            else:
                err = abs(armPosition - armHeightAtHighPosition)
                robot.motor[8] = 1 * ((tau_p_up * err) + (tau_i * cumErr))
                cumErr += 0.2 * err
                continue
        
    print("\n\n============= exiting bringArmToHighPosition() ===================\n\n")


def bringArmToLowPosition():

    print("\n\n============= entering bringArmToLowPosition() ===================\n\n")

    while True:      
        robot.update() # must never be forgotten
        _, _, sensors = robot.read()

        if len(sensors) == 4:
            _, armPosition, _, _ = sensors

            print(f"current armPostion: {armPosition}")

            # move arm to bottom (reset) position
            if armPosition <= armHeightAtLowPosition * 1.05:
                robot.motor[8] = 0
                break
            else:
                err = abs(armPosition - armHeightAtLowPosition)
                r1 = -1 * (tau_p_down * err) * 0.8
                r2 = 0
                # counter the fall due to gravity
                if armPosition <= (armHeightAtLowPosition + armHeightAtHighPosition) / 2:
                    r1 = 7

                robot.motor[8] = r1 + r2

                continue
    
    print("\n\n============= exiting bringArmToLowPosition() ===================\n\n")

def openClaw():

    print("\n\n============= entering openClaw() ===================\n\n")

    for _ in range(15):
        robot.update() # must never be forgotten
        robot.motor[7] = 18
    
    robot.motor[7] = 0
    time.sleep(1)

    print("\n\n============= exiting openClaw() ===================\n\n")

def closeClaw():

    print("\n\n============= entering closeClaw() ===================\n\n")

    for _ in range(15):
        robot.update() # must never be forgotten
        robot.motor[7] = -18
    
    robot.motor[7] = 0
    time.sleep(1)

    print("\n\n============= exiting closeClaw() ===================\n\n")

def isBallInTheImage(image):
    return True

def verifyArmAndClawMovements():

    bringArmToHighPosition()
    time.sleep(1)
    openClaw()
    time.sleep(1)
    closeClaw()
    time.sleep(1)
    bringArmToLowPosition()
    time.sleep(1)
    openClaw()
    time.sleep(1)
    closeClaw()
    time.sleep(1)

fx = 460.92495728
fy = 460.85058594
cx = 315.10949707
cy = 176.72598267
h, w = (360, 640)
U = np.tile(np.arange(w).reshape((1, w)), (h, 1))
V = np.tile(np.arange(h).reshape((h, 1)), (1, w))
U = (U - cx) / fx
V = (V - cy) / fy

def get_XYZ(depth_image):
    Z = depth_image
    X = U * Z
    Y = V * Z
    # formatting magic
    XYZ = np.concatenate((
        X.reshape((-1, 1)),
        Y.reshape((-1, 1)),
        Z.reshape((-1, 1))
    ), axis=-1)
    return XYZ

if __name__ == "__main__":
    robot = RemoteInterface("192.168.1.156") # home
    # robot = RemoteInterface("192.168.50.194") # clbairobotics

    _iter = 0

    armHeightAtLowPosition = 483
    armHeightAtHighPosition = 1487
    tau_p_up = 0.2
    tau_p_down = 0.1
    tau_i = 0 #tau_p_up/60

    # make sure the arm moves as intended
    # verifyArmAndClawMovements()

    # object detection model
    model = maskrcnn_resnet50_fpn(pretrained=True, pretrained_backbone=True)
    model.eval()
    model.to('cuda')

    prev_rgbd_image = None
    option = open3d.pipelines.odometry.OdometryOption()
    # Intel Realsense D415 Intrinsic Parameters
    fx = 460.92495728   # FOV(x) -> depth2xyz -> focal length (x)
    fy = 460.85058594   # FOV(y) -> depth2xyz -> focal length (y)
    cx = 315.10949707   # 640 (width) 320
    cy = 176.72598267   # 360 (height) 180
    cam_intrinsic_params = open3d.camera.PinholeCameraIntrinsic(640, 360, fx, fy, cx, cy)
    camera_params = ( fx, fy, cx, cy )
    at_detector = Detector(families='tag16h5',
                            nthreads=1,
                            quad_decimate=1.0,
                            quad_sigma=0.0,
                            refine_edges=1,
                            decode_sharpening=0.25,
                            debug=0)

    global_T = np.identity(4)

    debug = True

    while True:
        _iter += 1

        print(f"\n\n======== iteration: {_iter} =======================\n\n")

        robot.update() # must never be forgotten
        color, depth, sensors = robot.read()
        
        preprocess = transforms.Compose([ transforms.ToTensor(), ])
        input_tensor = preprocess(Image.fromarray(color))
        input_batch = input_tensor.unsqueeze(0).to('cuda')
        with torch.no_grad():
            output = model(input_batch)[0]
        output = {l: output[l].to('cpu').numpy() for l in output}

        object_index = 37
        indeces_found = [i for i in range(len(output["labels"])) if \
            (output["labels"][i] == object_index and output["scores"][i] > 0)]
        scores = output["scores"][indeces_found] # between 0 and 1, 1 being most confident
        labels = output["labels"][indeces_found] # integer id of the object found, refer to COCO classes
        masks  = output["masks"] [indeces_found] # mask in the image (N, 1, height, width)
        boxes  = output["boxes"] [indeces_found] # bounding boxes, not really needed

        print(f"\nbox around ball(s) location(s): {boxes}\n")

        # get a single mask's centered XYZ coordinates, ball's location
        single_mask = np.zeros((360, 640), dtype=np.uint8)
        if len(masks) > 0:
            single_mask = masks[0].reshape((360, 640))
        ball_depth = depth * (single_mask > 0)
        xyz = get_XYZ(ball_depth)

        print(f"\nxyz of the ball: {xyz}\n")

        num_pixels = np.sum(ball_depth > 0)
        if num_pixels > 0:
            average_xyz = np.sum(xyz, axis=0) / num_pixels

        if num_pixels > 0:
            print(f"\ncenter of the ball location: {average_xyz}\n")

        cv2.imshow("color", color)
        cv2.waitKey(1)

        # =========== slam ===========================

        # filter the depth image so that noise is removed
        depth = depth.astype(np.float32) / 1000.
        mask = np.bitwise_and(depth > 0.1, depth < 3.0) # -> valid depth points
        filtered_depth = np.where(mask, depth, 0)

        # converting to Open3D's format so we can do odometry
        o3d_color = open3d.geometry.Image(color)
        o3d_depth = open3d.geometry.Image(filtered_depth)
        o3d_rgbd  = open3d.geometry.RGBDImage.create_from_color_and_depth(o3d_color, o3d_depth)

        tags = at_detector.detect(
        cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), True, camera_params, 2.5)
        found_tag = False
        for tag in tags:
            if tag.decision_margin < 50: continue
            #found_tag!
            if tag.tag_id == 1: # 1..8
                print(tag.pose_R, tag.pose_t)
                # global_T = get_pose(tag.pose_R, tag.pose_t) # use your get_pose algorithm here!
                found_tag = True
        
        if not found_tag and prev_rgbd_image is not None: # use RGBD odometry relative transform to estimate pose
            T = np.identity(4)
            ret, T, _ = open3d.pipelines.odometry.compute_rgbd_odometry(
                o3d_rgbd, prev_rgbd_image, cam_intrinsic_params, T,
                open3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option)
            global_T = global_T.dot(T)
            rotation = global_T[:3,:3]
            print(f"\nRotation: {R.from_matrix(rotation).as_rotvec(degrees=True)}")

        prev_rgbd_image = o3d_rgbd # we forgot this last time!

        # dont need this, but helpful to visualize
        filtered_color = np.where(np.tile(mask.reshape(360, 640, 1), (1, 1, 3)), color, 0)
        cv2.imshow("color", filtered_color)
        cv2.waitKey(1)











        for event in pygame.event.get():
            
            # if event.type == QUIT:
            #     pygame.quit()
            #     sys.exit()

            if event.type == KEYUP:
                robot.motor[0] = 0
                robot.motor[9] = 0
                robot.motor[8] = 0
                robot.motor[7] = 0

            elif event.type == KEYDOWN:

                # claw open
                if event.key == ord('e'):
                    robot.motor[7] = 20

                # claw close
                if event.key == ord('c'):
                    robot.motor[7] = -20



                # arm up
                if event.key == ord('q'):
                    robot.motor[8] = 100

                # arm down
                if event.key == ord('z'):
                    robot.motor[8] = -50



                # forward
                if event.key == ord('w'):
                    robot.motor[0] = -80         # left wheel
                    robot.motor[9] = 80          # right wheel

                # backward
                if event.key == ord('s'):
                    robot.motor[0] = 80         
                    robot.motor[9] = -80          

                # left
                if event.key == ord('a'):
                    robot.motor[0] = 0         
                    robot.motor[9] = 80          

                # right
                if event.key == ord('d'):
                    robot.motor[0] = -80        
                    robot.motor[9] = 0          


        # print("Color: ")
        # print("\ttype --> ", type(color));
        # print("\tsize --> ", color.size);
        # print("\tdim --> ", color.ndim);
        # print("\trow-dim --> ", color[0].size);
        # # print("\tvalue --> ", color);

        # print("Depth: ")
        # print("\ttype --> ", type(depth));
        # print("\tsize --> ", depth.size);
        # print("\tdim --> ", depth.ndim);
        # print("\trow-dim --> ", depth[0].size);
        # # print("\tvalue --> ", depth);

        #break

    robot.__del__()




