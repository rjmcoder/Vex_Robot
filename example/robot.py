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
import random

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

def update_robot_goto(robot, state, goal):
    dpos = np.array(goal) - state[:2]
    dist = np.sqrt(dpos[0] ** 2 + dpos[1] ** 2)
    theta = np.degrees(np.arctan2(dpos[1], dpos[0])) - state[2]
    theta = (theta + 180) % 360 - 180 # [-180, 180]
    Pforward = 30
    Ptheta = 8

    # restrict operating range
    if np.abs(theta) < 30:
        #   # P-controller
        robot.motor[0] = -Pforward * dist + Ptheta * theta
        robot.motor[9] =  Pforward * dist + Ptheta * theta
    else:
        robot.motor[0] = 127 if theta > 0 else -127
        robot.motor[9] = 127 if theta > 0 else -127

def update_robot_move_arm(robot, angle, goal):
    # P-controller with constant current
    robot.motor[1] = (goal - angle) * 127 + 30

def find_path(objmap, state, goaltag):
  visited = np.zeros(objmap.shape, np.uint8)
  frontier = [(int(state[0]) + 72, int(state[1]) + 72)]
  parent = {}

  while frontier:
    curr = frontier.pop(0)
    visited[curr[0], curr[1]] = True
    if objmap[curr[0], curr[1]] == goaltag:
      path = [curr]
      while curr in parent:
        path.insert(0, parent[curr])
        curr = parent[curr]
      for i in range(len(path)):
        path[i] = (path[i][0] - 72, path[i][1] - 72)
      return path
    neighbors = [
      (curr[0] - 1, curr[1] + 0),
      (curr[0] + 1, curr[1] + 0),
      (curr[0] + 0, curr[1] - 1),
      (curr[0] + 0, curr[1] + 1)
    ]
    for neighbor in neighbors:
      if 0 <= neighbor[0] < 144 and \
         0 <= neighbor[1] < 144 and \
         not visited[neighbor[0], neighbor[1]]:
        frontier.append(neighbor)
        visited[neighbor[0], neighbor[1]] = True
        parent[neighbor] = curr

  return None

def draw_boxes(boxes, classes, labels, image):
    """
    Draws the bounding box around a detected object.
    """
    lw = max(round(sum(image.shape) / 2 * 0.003), 2)  # Line width.
    tf = max(lw - 1, 1) # Font thickness.
    for i, box in enumerate(boxes):
        # color = COLORS[labels[i]]
        cv2.rectangle(
            img=image,
            pt1=(int(box[0]), int(box[1])),
            pt2=(int(box[2]), int(box[3])),
            color=(0,0,255), #color[::-1],
            thickness=lw
        )
        cv2.putText(
            img=image,
            text="ball",
            org=(int(box[0]), int(box[1]-5)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=lw / 3,
            color=(0,0,255), #color[::-1],
            thickness=tf,
            lineType=cv2.LINE_AA
        )
    return image

def wheelsMoveForward(x):
    robot.motor[0] = -x  # left wheel
    robot.motor[9] = x  # right wheel

def wheelsMoveBackward(x):
    robot.motor[0] = x  # left wheel
    robot.motor[9] = -x  # right wheel

def wheelsTurnLeft(x):
    robot.motor[0] = 0
    robot.motor[9] = x

def wheelsTurnSharpLeft(x):
    robot.motor[0] = x
    robot.motor[9] = x

def wheelsTurnRight(x):
    robot.motor[0] = -x
    robot.motor[9] = 0

def wheelsTurnSharpRight(x):
    robot.motor[0] = -x
    robot.motor[9] = -x

if __name__ == "__main__":
    robot = RemoteInterface("192.168.1.156") # home
    # robot = RemoteInterface("192.168.50.194") # clbairobotics

    _iter = 0

    centerOfClawInImage = (320, 186)

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

        # =========== object detection ====================================================================

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
        boxes = boxes[scores >= 0.5] #.astype(np.int32)
        masks = masks[scores >= 0.5] #.astype(np.int32)

        boxes = np.sort(boxes, axis = 0)

        print(f"\nbox around ball(s) location(s): {boxes}\n")
        # print(f"\nmask around ball(s) location(s): {masks}\n")

        color = draw_boxes(boxes, ["sports ball"], labels, color)

        if len(boxes) != 0:
            print("found ball(s)")
            firstBall = boxes[0]
            x1, y1, x2, y2 = firstBall
            centerOfBall = (int((x1+x2)/2), int((y1+y2)/2))
            cx, cy = centerOfBall
            cv2.circle(color, (int(centerOfBall[0]), int(centerOfBall[1])), 2, (0, 0, 255), -1)
            print(f"\ncenter of the ball location: {centerOfBall}\n")
            cv2.imshow("color", color)
            cv2.waitKey(1)

            # check if ball is within the width of the claw
            if centerOfClawInImage[0] * 0.95 < cx < centerOfClawInImage[0] * 1.05:
                print("ball within claws width")
                if cy > 250:
                    print("reached close to the ball")
                    # while True:
                    #     if limitSwitchPressed == True:
                    #         wheelsMoveForward(0)
                    # for i in range(5):
                    #     wheelsMoveForward(10)
                    wheelsMoveForward(0)
                else:
                    wheelsMoveForward(50)
            else:
                print("turning to align towards the ball")
                # ball is towards the right
                if cx > centerOfClawInImage[0] * 1.05:
                    wheelsTurnRight(50)
                    continue
                # ball is towards the left
                if cx < centerOfClawInImage[0] * 0.95:
                    wheelsTurnLeft(50)
                    continue
        else:
            # move the wheels and look at some other location for a ball
            print("looking for a ball")
            randomChoice = random.choice([1,2,3,4])
            power = 50
            steps = 20
            if randomChoice == 1:
                for i in range(steps):
                    wheelsMoveForward(power)
            if randomChoice == 2:
                for i in range(steps):
                    wheelsMoveBackward(power)
            if randomChoice == 3:
                for i in range(steps*5):
                    wheelsTurnSharpRight(power)
            if randomChoice == 4:
                for i in range(steps*5):
                    wheelsTurnSharpLeft(power)
            continue

        # # get a single mask's centered XYZ coordinates, ball's location
        # single_mask = np.zeros((360, 640), dtype=np.uint8)
        # if len(masks) > 0:
        #     single_mask = masks[0].reshape((360, 640))
        # ball_depth = depth * (single_mask > 0)
        # xyz = get_XYZ(ball_depth)

        # print(f"\nxyz of the ball: {xyz}\n")

        # num_pixels = np.sum(ball_depth > 0)
        # if num_pixels > 0:
        #     average_xyz = np.sum(xyz, axis=0) / num_pixels
        #     print(f"\ncenter of the ball location: {average_xyz}\n")


        cv2.imshow("color", color)
        cv2.waitKey(1)
        pass

        # # =========== slam ====================================================================
        #
        # # filter the depth image so that noise is removed
        # depth = depth.astype(np.float32) / 1000.
        # mask = np.bitwise_and(depth > 0.1, depth < 3.0) # -> valid depth points
        # filtered_depth = np.where(mask, depth, 0)
        #
        # cv2.imshow("filtered_depth", filtered_depth)
        # cv2.waitKey(1)
        #
        # # converting to Open3D's format so we can do odometry
        # o3d_color = open3d.geometry.Image(color)
        # o3d_depth = open3d.geometry.Image(filtered_depth)
        # o3d_rgbd  = open3d.geometry.RGBDImage.create_from_color_and_depth(o3d_color, o3d_depth)
        #
        # tags = at_detector.detect(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), True, camera_params, 2.5)
        # found_tag = False
        # for tag in tags:
        #     if tag.decision_margin < 50: continue
        #     #found_tag!
        #     if tag.tag_id == 1: # 1..8
        #         print(tag.pose_R, tag.pose_t)
        #         # global_T = get_pose(tag.pose_R, tag.pose_t) # use your get_pose algorithm here!
        #         found_tag = True
        #
        # if not found_tag and prev_rgbd_image is not None: # use RGBD odometry relative transform to estimate pose
        #     T = np.identity(4)
        #     ret, T, _ = open3d.pipelines.odometry.compute_rgbd_odometry(
        #         o3d_rgbd, prev_rgbd_image, cam_intrinsic_params, T,
        #         open3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option)
        #     global_T = global_T.dot(T)
        #     rotation = global_T[:3,:3]
        #     print(f"\nRotation: {R.from_matrix(rotation).as_rotvec(degrees=True)}")
        #
        # prev_rgbd_image = o3d_rgbd # we forgot this last time!
        #
        # # dont need this, but helpful to visualize
        # filtered_color = np.where(np.tile(mask.reshape(360, 640, 1), (1, 1, 3)), color, 0)
        # cv2.imshow("color", filtered_color)
        # cv2.waitKey(1)
        #
        #
        # # ==================== path planning ==============================
        # # x, y, theta = RGBDOdometry()
        # sensors = robot.read()
        # x, y = robot.pos
        # theta = robot.angle  # get these from SLAM
        #
        # path = find_path(objective_map, (x, y, theta), 1)
        # goal = path[min(5, len(path) - 1)]  # get 5 steps ahead
        # update_robot_goto(robot, (x, y, theta), goal)








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

    # robot.__del__()




