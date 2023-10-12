clear;clc;close all;

ur10 = loadrobot("universalUR10");

% for i = 1:4
%     subplot(2,2,i)
%     config = randomConfiguration(ur10);
%     show(ur10,config);
% end

interactiveGUI = interactiveRigidBodyTree(ur10);
