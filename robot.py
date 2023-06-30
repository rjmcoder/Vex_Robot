from cortano import RemoteInterface

if __name__ == "__main__":
  robot = RemoteInterface("10.0.0.53")
  while True:
    forward = robot.keys["w"] - robot.keys["s"]
    robot.motor[0] = forward * 64
    robot.motor[9] = forward * 64
    robot.update()