%% Human-like code for MATLAB
clear;clc;close all;
%% Load the robot model
robot = loadrobot("universalUR3");
T = [1 0 0 0;0 0.7071 0.7071 0;0 -0.7071 0.7071 0;0 0 0 1];
HombroLim = [-3.14,3.14];
HombroLim2 = [-2, 1.5];
% Body values
weights = [2, 3.42, 1.26, 0.8, 0.8, 0.35];
configSoln = robot.homeConfiguration;
%% Collision environment
robotModel = loadrobot("universalUR3","DataFormat","column");
T1 = [1 0 0 0.07;0 0.7071 0.7071 0.13;0 -0.7071 0.7071 1.15;0 0 0 1];
setFixedTransform(robotModel.Bodies{1,1}.Joint,T1);
ss = manipulatorStateSpace(robotModel);
sv = manipulatorCollisionBodyValidator(ss, SkippedSelfCollisions="parent");
sv.ValidationDistance = 0.1;
sv.IgnoreSelfCollision = true;

% Load the body as a collision object
cagedata=stlread('body1.stl');
body1 = collisionMesh(cagedata.Points./1000);
% body1.Vertices(:) = body1.Vertices(:)*1.03;
cagedata=stlread('body2.stl');
body2 = collisionMesh(cagedata.Points./1000);
% body2.Vertices(:) = body2.Vertices(:)*1.03;
cagedata=stlread('body3.stl');
body3 = collisionMesh(cagedata.Points./1000);
% body3.Vertices(:) = body3.Vertices(:)*1.03;
cagedata=stlread('body4.stl');
body4 = collisionMesh(cagedata.Points./1000);
% body4.Vertices(:) = body4.Vertices(:)*1.03;
matrizRot = [0 -1 0 0;1 0 0 0;0 0 1 -0.03;0 0 0 1]; %(0.05248825/2)-0.025
body1.Pose = matrizRot;
body2.Pose = matrizRot;
body3.Pose = matrizRot;
body4.Pose = matrizRot;
env = {body1 body2 body3 body4};
sv.Environment = env;
% To visualized the environment
% figure(11)
% show(robotModel,"Collisions","off");
% hold on
% show(env{1});
% show(env{2});
% show(env{3});
% show(env{4});
% hold off


%% Inverse kinematics solver
%% Read the files

%path = '/home/carlos/TAICHI/HumanData/Prueba1/DatosBrazoHumano.csv';
%path2 = '/home/carlos/TAICHI/HumanData/Prueba1/CodoHumano.csv';

path = '/home/carlos/TAICHI_Carlos/HumanData/PruebaBrazoAntiguo/DatosBrazoHumano.csv';
path2 = '/home/carlos/TAICHI_Carlos/HumanData/PruebaBrazoAntiguo/CodoHumano.csv';

%path = '/home/carlos/TAICHI_Carlos/HumanData/Prueba32/DatosBrazoIzquierdo.csv';
%path2 = '/home/carlos/TAICHI_Carlos/HumanData/Prueba32/CodoIzquierdo.csv';

matrixRead = readmatrix(path);
CodoRead = readmatrix(path2);
CodoOrganizado = [];
MEJORES = [];
PEORES = [];
iter = length(matrixRead);
iter2 = length(CodoRead);
k=0;
configFinal = [];
almacenamiento = [];
RES=[];
ROTMAT = [];
VecCODO = [];
CODRob = [];
% Organised the elbow values
for j=1:3:iter2
    CodoOrganizado = [CodoOrganizado; CodoRead(j,1) CodoRead(j+1) CodoRead(j+2)];
end
%% Main 
% Weigths values
W_rax = 1; %Position weigth
W_rao = 10; %Orientation weigth
W_A = 50; %Humanity weigth
%Evaluation loop
gola = []; %Goal Point with respect to human shoulder CS
gola2 = []; %Goal Point with respect to human shoulder CS rotated 45
matrixCodo = zeros(4,4);
for i=1:4:iter
    k = k+1;
    DistDif = 10000000;
    DistDigOG = DistDif;
    iteration = 0;
    check = true;
    
    % End efector matrix
    matrix = [matrixRead(i,1) matrixRead(i,2) matrixRead(i,3) matrixRead(i,4);
        matrixRead(i+1,1) matrixRead(i+1,2) matrixRead(i+1,3) matrixRead(i+1,4);
        matrixRead(i+2,1) matrixRead(i+2,2) matrixRead(i+2,3) matrixRead(i+2,4);
        matrixRead(i+3,1) matrixRead(i+3,2) matrixRead(i+3,3) matrixRead(i+3,4)];
    matrix2 = inv(T) * matrix;
    matrix(1:3,1:3)  = rotz(180) * matrix(1:3,1:3);
    matrix(1:2,4) = -matrix(1:2,4);
    matrix = T * matrix;
    
    % Elbow matrix
    VectorCodo = [CodoOrganizado(k,1), CodoOrganizado(k,2),CodoOrganizado(k,3)];
    VecCODO = [VecCODO;VectorCodo];
    X_RAGoal = [matrix2(1,4) matrix2(2,4) matrix2(3,4)];
    gola = [gola;[matrixRead(i,4) matrixRead(i+1,4) matrixRead(i+2,4)]];
    gola2 = [gola2;X_RAGoal];
    Rot_mat = [matrix2(1,1) matrix2(1,2) matrix2(1,3);
        matrix2(2,1) matrix2(2,2) matrix2(2,3);
        matrix2(3,1) matrix2(3,2) matrix2(3,3)];
    EulerAngles = rotm2eul(Rot_mat);
    quat_h = eul2quat(EulerAngles);
    
    disp("------------- NEW ITERATION -------------")
    disp(k)
   
    % IK using UR3_Inverse_Kinematics
    try
        IK_UR

        for ii = 1:1:8
            for jj = 1:1:6
                configSoln(jj).JointPosition =res(ii,jj);
            end
            % Check collision using the exact collision model
            goalConfig = [configSoln(1).JointPosition configSoln(2).JointPosition configSoln(3).JointPosition configSoln(4).JointPosition configSoln(5).JointPosition configSoln(6).JointPosition];
            [validState,~] = checkCollision(robotModel,goalConfig',env,"IgnoreSelfCollision","off","Exhaustive","on","SkippedSelfCollisions","parent");
            if ~any(validState)

                % End efector for the specific configuration
                EfectorFinal = getTransform(robot,configSoln,'tool0','base_link');
                X_RA = [EfectorFinal(1,4) EfectorFinal(2,4) EfectorFinal(3,4)];
                Rot_EF =[EfectorFinal(1,1) EfectorFinal(1,2) EfectorFinal(1,3);
                    EfectorFinal(2,1) EfectorFinal(2,2) EfectorFinal(2,3);
                    EfectorFinal(3,1) EfectorFinal(3,2) EfectorFinal(3,3)];
                EulerAnglesEF = rotm2eul(Rot_EF);
                quat_r = eul2quat(EulerAnglesEF);

                % Detect if the point is in the limits'
                %pointEnd = [gola(k,1)+0.07,gola(k,2)+0.13,gola(k,3)+1.15];
                %checkConfig

                % Wrist 2
                Wrist2 = getTransform(robot,configSoln,'wrist_2_link','base_link');
                S_wrist2 = [Wrist2(1,4) Wrist2(2,4) Wrist2(3,4)];

                % Wrist 1
                Wrist1 = getTransform(robot,configSoln,'wrist_1_link','base_link');
                S_wrist1 = [Wrist1(1,4) Wrist1(2,4) Wrist1(3,4)];

                % Wrist 3
                Wrist3 = getTransform(robot,configSoln,'wrist_3_link','base_link');
                S_wrist3 = [Wrist3(1,4) Wrist3(2,4) Wrist3(3,4)];

                %Elbow
                Codo = getTransform(robot,configSoln,'forearm_link','base_link');
                S_codo = [Codo(1,4) Codo(2,4) Codo(3,4)];

                %Shoulder
                HombroSaliente = getTransform(robot,configSoln,'shoulder_link','base_link');
                S_hombro = [HombroSaliente(1,4) HombroSaliente(2,4) HombroSaliente(3,4)];

                %Calculate the diference in position
                d_RAx = distPosition(X_RA,X_RAGoal);
                % Calculate the diference in orientation
                d_RAo = distOrientation(quat_r,quat_h);
                % Calculate the diference between the human elbow and the
                % rest of the arm
                MDistancia = distanceMetric(VectorCodo, S_wrist2,S_wrist1,S_wrist3,S_codo,S_hombro,gola');
               
                %Wrist error estimation
                if i == 1 || ~exist('Wrist_old','var')
                    ErroWrist = 0;
                else
                    ErroWrist = variationWrist(configSoln(4).JointPosition,Wrist_old);
                end
                DistFinal = real(W_rax*d_RAx + W_rao*d_RAo + W_A*MDistancia + ErroWrist);
                
                CodoRobotrot = rotx(45)*S_codo';
                WristRobotRot = rotx(45)*S_wrist1';

                % Check if it is finished
                if d_RAx<=0.05 && d_RAo <= 0.16 && ~(CodoRobotrot(2)<-0.1 && WristRobotRot(2)>CodoRobotrot(2)) && DistFinal < DistDif && configSoln(1).JointPosition >= HombroLim(1) && configSoln(1).JointPosition <= HombroLim(2) && configSoln(2).JointPosition >= HombroLim2(1) && configSoln(2).JointPosition <= HombroLim2(2)
                    DistDif = DistFinal;
                    MejorConfig = [configSoln(1).JointPosition configSoln(2).JointPosition configSoln(3).JointPosition configSoln(4).JointPosition configSoln(5).JointPosition configSoln(6).JointPosition];
                    mejor_ii = ii;
                end
            end
        end
        if DistDif ~= DistDigOG
            Wrist_old = MejorConfig(1,4);
            MEJORES = [MEJORES;mejor_ii];
            configFinal = [configFinal;MejorConfig];
            ROTMAT = [ROTMAT;Rot_mat];

        else
            disp('Not valid configuration found')
            worst = k;
            PEORES = [PEORES;worst];
            dif
        end

    catch
        disp('Wrong IK value')
    end   
    
end
% Filter the incorrect data and correct the position values
gola(PEORES,:) = [];
gola2(PEORES,:) = [];
gola =[gola(:,1)+0.07,gola(:,2)+0.13,gola(:,3)+1.15];

%% Plot the movements of the robotics arm respect the human elbow and wrist
robot = loadrobot("universalUR3");
T = [1 0 0 0.07;0 0.7071 0.7071 0.13;0 -0.7071 0.7071 1.15;0 0 0 1];
setFixedTransform(robot.Bodies{1,1}.Joint,T);
configuraciones = robot.homeConfiguration;
k=0;
DISTANCIAS = [];
END = [];
ROTEND = [];
X =[];
Y = [];
Z = [];


for i=1:1:length(configFinal)
    k= k+1
    configuraciones(1).JointPosition = configFinal(i,1);
    configuraciones(2).JointPosition = configFinal(i,2);
    configuraciones(3).JointPosition = configFinal(i,3);
    configuraciones(4).JointPosition = configFinal(i,4);
    configuraciones(5).JointPosition = configFinal(i,5);
    configuraciones(6).JointPosition = configFinal(i,6);
    zoom(gca,'on');
    show(robot,configuraciones,"Collisions","off");
    view(90,0);
    CodoR = getTransform(robot,configuraciones,'forearm_link','base_link');
    CODRob = [CODRob;CodoR(1,4) CodoR(2,4) CodoR(3,4)];
    camzoom(1.7);
    hold on
    show(env{1});
    show(env{2});
    show(env{3});
    show(env{4});
    hold on
    plot3(VecCODO(k,1)+0.07,VecCODO(k,2)+0.13,VecCODO(k,3)+1.15,'o','Color','g','MarkerSize',10,'MarkerFaceColor','g')
    plot3(gola(k,1),gola(k,2),gola(k,3),'o','Color','r','MarkerSize',10,'MarkerFaceColor','r')
    hold on
    PuntoEnd = getTransform(robot,configuraciones,'tool0','base_link');
    Punto = [gola(k,1),gola(k,2),gola(k,3)];
    END = [END;PuntoEnd(1,4),PuntoEnd(2,4),PuntoEnd(3,4)];
    ROTEND = [ROTEND;PuntoEnd(1,1),PuntoEnd(1,2),PuntoEnd(1,3);PuntoEnd(2,1),PuntoEnd(2,2),PuntoEnd(2,3);PuntoEnd(3,1),PuntoEnd(3,2),PuntoEnd(3,3)];

    pause(0.001);
    hold off
end

%% Metrics

frechetIzq = frechet(gola2, END)
angularSimilarityIzq = angular_similarity(gola2, END)
jerkIzqHuman = jerk(gola2)
jerkIzqRobot = jerk(END)

%% Plot in 3D (smoothed to visualized it better)
figure;
plot3(gola2(:,1),gola2(:,2),gola2(:,3),'Color','g');
hold on
xlabel('X (m)')
ylabel('Y (m)')
zlabel('Z (m)')
plot3(END(:,1),END(:,2),END(:,3),'Color','b');
plot3(gola2(1,1),gola2(1,2),gola2(1,3),'o','Color','r','MarkerSize',8,'MarkerFaceColor','r');
plot3(gola2(length(gola2),1),gola2(length(gola2),2),gola2(length(gola2),3),'o','Color','m','MarkerSize',8,'MarkerFaceColor','m');
[t,s]=title("Comparision between data acquired");
t.FontSize = 16;
legend('Robot data','Human data','Initial point','Final point')
hold off

%% Plot the error values of the end efector to compare
PlotError



