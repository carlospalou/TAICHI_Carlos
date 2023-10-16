clear;clc;close all;

robot = loadrobot("quanserQArm",DataFormat="row");
figure(Visible="on")
show(robot);
xlim([-0.5 0.5])
ylim([-0.5 0.5])
zlim([-0.25 0.75])
hold on

ss = manipulatorStateSpace(robot);
sv = manipulatorCollisionBodyValidator(ss,SkippedSelfCollisions="parent");