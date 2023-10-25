
M = [1 0 0 0;0 1 0 1;0 0 1 0;0 0 0 1];
T = M*TI;
setFixedTransform(robotIzq.Bodies{1,1}.Joint,T);
figure(1)
show(robotIzq,"Collisions","off");
% hold on
% %show(robotModelDer,"Collisions","off");
% show(env{1});
% show(env{2});
% show(env{3});
% show(env{4});
% hold off
EfectorFinalIzq = getTransform(robotIzq,configSolnIzq,'tool0','base_link');