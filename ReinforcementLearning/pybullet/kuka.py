import pybullet as p
import time

p.connect(p.GUI)

objects = [
    p.loadURDF("kuka.urdf")
]
kuka = objects[0]
jointPositions = [-0.000000, -0.000000, 0.000000, 1.570793, 0.000000, -1.036725, 0.000001, .0, .0]

for jointIndex in range(p.getNumJoints(kuka)):
  p.resetJointState(kuka, jointIndex, jointPositions[jointIndex])
  p.setJointMotorControl2(kuka, jointIndex, p.POSITION_CONTROL, jointPositions[jointIndex], 0)

p.setRealTimeSimulation(1)
ref_time = time.time()

running_time = 60  # seconds
while (time.time() < ref_time + running_time):
  #p.setGravity(0, 0, -10)
  #p.stepSimulation()
  pass

p.disconnect()