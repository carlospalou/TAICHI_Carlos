%% Human-like code for MATLAB
clear;clc;close all;

%% Load the robot model
robot = loadrobot("universalUR3"); % Usa una funcion de Robotics System Toolbox para cargar el robot UR3
T = [1 0 0 0;0 0.7071 0.7071 0;0 -0.7071 0.7071 0;0 0 0 1];
HombroLim = [-3.14,3.14];
HombroLim2 = [-2, 1.5];
% Body values
weights = [2, 3.42, 1.26, 0.8, 0.8, 0.35];
configSoln = robot.homeConfiguration; % La configuracion predeterminada del robot

%% Collision environment 
robotModel = loadrobot("universalUR3","DataFormat","column"); % Para utilizar vectores columna en las configuraciones de articulacion
ss = manipulatorStateSpace(robotModel); % Se crea un espacio de estados para representar y manipular el espacio de configuraciones del robot
sv = manipulatorCollisionBodyValidator(ss, SkippedSelfCollisions="parent"); % Se crea un validador de colisiones para el espacio de estados
sv.ValidationDistance = 0.1; % Distancia validacion
sv.IgnoreSelfCollision = true; % Ignora el propio brazo como colision (no el resto del robot)

% Load the body as a collision object
cagedata = stlread('body1.stl'); 
body1 = collisionMesh(cagedata.Points./1000); % Crea un objeto de colision tridimensional escalando los datos leidos de body1.stl
%body1.Vertices(:) = body1.Vertices(:)*1.03; % Multiplica las coordenadas de los vertices por 1.03, escalando el objeto
cagedata = stlread('body2.stl');
body2 = collisionMesh(cagedata.Points./1000);
% body2.Vertices(:) = body2.Vertices(:)*1.03;
cagedata = stlread('body3.stl');
body3 = collisionMesh(cagedata.Points./1000);
% body3.Vertices(:) = body3.Vertices(:)*1.03;
cagedata = stlread('body4.stl');
body4 = collisionMesh(cagedata.Points./1000);
% body4.Vertices(:) = body4.Vertices(:)*1.03;

matrizRot = [0 -1 0 0;1 0 0 0;0 0 1 -0.03;0 0 0 1]; %(0.05248825/2)-0.025 % Matriz de rotacion para los elementos del cuerpo
body1.Pose = matrizRot; % Transformacion de pose para las partes del cuerpo
body2.Pose = matrizRot;
body3.Pose = matrizRot;
body4.Pose = matrizRot;
env = {body1 body2 body3 body4}; % Crea un entorno
sv.Environment = env; % Utiliza el validador de colision sv1 en el entorno env

T1 = [1 0 0 0.07;0 0.7071 0.7071 0.13;0 -0.7071 0.7071 1.15;0 0 0 1];
T2 = [-1 0 0 0.07;0 -0.707 -0.707 -0.13;0 -0.707 0.707 1.15;0 0 0 1];


%% Inverse kinematics solver

% Read the files
path_izq = '/home/carlos/TAICHI_Carlos/HumanData/Prueba1/DatosBrazoIzquierdo.csv';
path2_izq = '/home/carlos/TAICHI_Carlos/HumanData/Prueba1/CodoIzquierdo.csv';
path_der = '/home/carlos/TAICHI_Carlos/HumanData/Prueba1/DatosBrazoDerecho.csv';
path2_der = '/home/carlos/TAICHI_Carlos/HumanData/Prueba1/CodoDerecho.csv';

MatrixIzqRead = readmatrix(path_izq); % Almacena los datos del .csv en una matrix
CodoIzqRead = readmatrix(path2_izq);
MatrixDerRead = readmatrix(path_der);
CodoDerRead = readmatrix(path2_der);

iter_izq = length(MatrixIzqRead); % Cantidad de elementos de la matriz
iter2_izq = length(CodoIzqRead);
iter_der = length(MatrixDerRead);
iter2_der = length(CodoDerRead);

CodoIzqOrganizado = [];
CodoDerOrganizado = [];

MEJORES_IZQ = [];
PEORES_IZQ = [];
MEJORES_DER = [];
PEORES_DER = [];
ConfigFinalIzq= [];
ConfigFinalDer= [];
AlmacenamientoIzq = [];
AlmacenamientoDer = [];
RES_IZQ=[];
RES_DER=[];
ROTMAT_IZQ = [];
ROTMAT_DER = [];
VEC_CODO_IZQ = [];
VEC_CODO_DER = [];
CODRob_IZQ = [];
CODRob_DER = [];

% Organised the elbow values
for j=1:3:iter2_izq  
    CodoIzqOrganizado = [CodoIzqOrganizado; CodoIzqRead(j,1) CodoIzqRead(j+1) CodoIzqRead(j+2)];
end

for j=1:3:iter2_der  
    CodoDerOrganizado = [CodoDerOrganizado; CodoDerRead(j,1) CodoDerRead(j+1) CodoDerRead(j+2)];
end


%% Main 

% Weigths values
W_rax = 1; % Position weigth
W_rao = 10; % Orientation weigth
W_A = 50; % Humanity weigth

% Evaluation loop
Goal_Izq = []; % Goal Point with respect to human shoulder CS
Goal2_Izq = []; % Goal Point with respect to human shoulder CS rotated 45
MatrixCodoIzq = zeros(4,4);
Goal_Der = [];
Goal2_Der = [];
MatrixCodoDer = zeros(4,4);

k=0;

for i=1:4:iter_izq 
    k = k+1;
    DistDif = 10000000;
    DistDigOG = DistDif;
    iteration = 0;
    check = true;
    
    % Elbow matrix
    VectorCodoIzq = [CodoIzqOrganizado(k,1), CodoIzqOrganizado(k,2), CodoIzqOrganizado(k,3)];
    VEC_CODO_IZQ = [VEC_CODO_IZQ; VectorCodoIzq];


    % End efector matrix
    MatrixIzq = [MatrixIzqRead(i,1) MatrixIzqRead(i,2) MatrixIzqRead(i,3) MatrixIzqRead(i,4); % Orientacion del EF obtenida de los datos de la camara
        MatrixIzqRead(i+1,1) MatrixIzqRead(i+1,2) MatrixIzqRead(i+1,3) MatrixIzqRead(i+1,4);
        MatrixIzqRead(i+2,1) MatrixIzqRead(i+2,2) MatrixIzqRead(i+2,3) MatrixIzqRead(i+2,4);
        MatrixIzqRead(i+3,1) MatrixIzqRead(i+3,2) MatrixIzqRead(i+3,3) MatrixIzqRead(i+3,4)];
    MatrixIzq2 = inv(T) * MatrixIzq;
    MatrixIzq(1:3,1:3)  = rotz(180) * MatrixIzq(1:3,1:3); 
    MatrixIzq(1:2,4) = -MatrixIzq(1:2,4);
    MatrixIzq = T * MatrixIzq; 

    matrix = MatrixIzq; % Para IK_UR

    X_RA_Goal_Izq = [MatrixIzq2(1,4) MatrixIzq2(2,4) MatrixIzq2(3,4)]; % Posicion del EF obtenida de los datos de la camara
    Goal_Izq = [Goal_Izq; [MatrixIzqRead(i,4) MatrixIzqRead(i+1,4) MatrixIzqRead(i+2,4)]]; % Va almacenando las orientaciones
    Goal2_Izq = [Goal2_Izq;X_RA_Goal_Izq]; % Va almacenando las posiciones
    RotMatIzq = [MatrixIzq2(1,1) MatrixIzq2(1,2) MatrixIzq2(1,3); % Matriz de rotacion del EF en funcion de los datos de la camara
        MatrixIzq2(2,1) MatrixIzq2(2,2) MatrixIzq2(2,3);
        MatrixIzq2(3,1) MatrixIzq2(3,2) MatrixIzq2(3,3)];
    EulerAnglesIzq = rotm2eul(RotMatIzq); % Pasa de matriz de rotacion a angulos de Euler
    Quat_h_izq = eul2quat(EulerAnglesIzq); % Pasa de angulos de Euler a cuaterniones
    
    disp("------------- NEW LEFT ARM ITERATION -------------")
    disp(k)
   
    % IK using UR3_Inverse_Kinematics
    try
        IK_UR % Con MatrixIzq

        for ii = 1:1:8
            for jj = 1:1:6 % res son los resultados de UR3_Inverse_Kinematics, 8 soluciones con 6 articulaciones (?)
                configSol(jj).JointPosition = res(ii,jj); % Asigna la posicion de las articulaciones para cada configuracion
            end

            % Check collision using the collision model
            setFixedTransform(robotModel.Bodies{1,1}.Joint,T1);

            GoalConfigIzq = [configSol(1).JointPosition configSol(2).JointPosition configSol(3).JointPosition configSol(4).JointPosition configSol(5).JointPosition configSol(6).JointPosition];
            [validState,~] = checkCollision(robotModel,GoalConfigIzq',env,"IgnoreSelfCollision","off","Exhaustive","on","SkippedSelfCollisions","parent");
            if ~any(validState)

                % End efector for the specific configuration
                EfectorFinalIzq = getTransform(robot,configSol,'tool0','base_link'); % Orientacion del EF en funcion de la configuracion
                X_RA_Izq = [EfectorFinalIzq(1,4) EfectorFinalIzq(2,4) EfectorFinalIzq(3,4)]; % Posicion del EF en funcion de la configuracion
                Rot_EF_Izq =[EfectorFinalIzq(1,1) EfectorFinalIzq(1,2) EfectorFinalIzq(1,3); % Matriz de rotacion en funcion de la configuracion
                    EfectorFinalIzq(2,1) EfectorFinalIzq(2,2) EfectorFinalIzq(2,3);
                    EfectorFinalIzq(3,1) EfectorFinalIzq(3,2) EfectorFinalIzq(3,3)];
                EulerAnglesEFIzq = rotm2eul(Rot_EF_Izq); 
                Quat_r_izq = eul2quat(EulerAnglesEFIzq);


                % Wrist 1
                Wrist1Izq = getTransform(robot,configSol,'wrist_1_link','base_link');
                S_wrist1_izq = [Wrist1Izq(1,4) Wrist1Izq(2,4) Wrist1Izq(3,4)]; % Guada las coordenadas de la muñeca


                % Wrist 2
                Wrist2Izq = getTransform(robot,configSol,'wrist_2_link','base_link');
                S_wrist2_izq = [Wrist2Izq(1,4) Wrist2Izq(2,4) Wrist2Izq(3,4)];

                % Wrist 3
                Wrist3Izq = getTransform(robot,configSol,'wrist_3_link','base_link');
                S_wrist3_izq = [Wrist3Izq(1,4) Wrist3Izq(2,4) Wrist3Izq(3,4)];

                %Elbow
                CodoIzq = getTransform(robot,configSol,'forearm_link','base_link');
                S_codo_izq = [CodoIzq(1,4) CodoIzq(2,4) CodoIzq(3,4)];

                %Shoulder
                HombroSalienteIzq = getTransform(robot,configSol,'shoulder_link','base_link');
                S_hombro_izq = [HombroSalienteIzq(1,4) HombroSalienteIzq(2,4) HombroSalienteIzq(3,4)];

           
                d_RAx_izq = distPosition(X_RA_Izq,X_RA_Goal_Izq); % Calcula la diferencia entre la posicion del EF obtenida de los datos de la camara y de la solucion de IK_UR
                d_RAo_izq= distOrientation(Quat_r_izq,Quat_h_izq); % Calcula la diferencia de orientacion del EF
        
                MDistanciaIzq = distanceMetric(VectorCodoIzq, S_wrist2_izq, S_wrist1_izq, S_wrist3_izq, S_codo_izq, S_hombro_izq, Goal_Izq'); % (?)
               
                % Wrist error estimation
                if i == 1 || ~exist('WristOldIzq','var')
                    ErroWristIzq = 0; % Si es la primera iteracion o no hay un valor para la mejor configuracion el error es 0
                else
                    ErroWristIzq = variationWrist(configSol(4).JointPosition,WristOldIzq); % Calcula el error como la diferencia entre el nuevo valor de la muñeca y el antiguo
                end
                DistFinalIzq = real(W_rax*d_RAx_izq + W_rao*d_RAo_izq + W_A*MDistanciaIzq + ErroWrist);
                
                CodoRobotRotIzq = rotx(45)*S_codo_izq';
                WristRobotRotIzq= rotx(45)*S_wrist1_izq';

                % Check if it is finished
                if d_RAx_izq <= 0.05 && d_RAo_izq <= 0.16 && ~(CodoRobotRotIzq(2)<-0.1 && WristRobotRotIzq(2)>CodoRobotRotIzq(2)) && DistFinalIzq < DistDif && configSol(1).JointPosition >= HombroLim(1) && configSol(1).JointPosition <= HombroLim(2) && configSol(2).JointPosition >= HombroLim2(1) && configSol(2).JointPosition <= HombroLim2(2)
                    DistDif = DistFinalIzq;
                    MejorConfig = [configSol(1).JointPosition configSol(2).JointPosition configSol(3).JointPosition configSol(4).JointPosition configSol(5).JointPosition configSol(6).JointPosition];
                    mejor_ii = ii;
                end
            end
        end
        if DistDif ~= DistDigOG
            WristOldIzq = MejorConfig(1,4);
            MEJORES_IZQ = [MEJORES_IZQ;mejor_ii];
            ConfigFinalIzq = [ConfigFinalIzq;MejorConfig];
            ROTMAT_IZQ = [ROTMAT_IZQ;RotMatIzq];

        else
            disp('Not valid configuration found')
            WorstIzq = k;
            PEORES_IZQ = [PEORES_IZQ;WorstIzq];
            dif
        end

    catch
        disp('Wrong IK value')
    end   
    
end

k=0;

for i=1:4:iter_der
    k = k+1;
    DistDif = 10000000;
    DistDigOG = DistDif;
    iteration = 0;
    check = true;
    
    % Elbow matrix
    VectorCodoDer = [CodoDerOrganizado(k,1), CodoDerOrganizado(k,2), CodoDerOrganizado(k,3)];
    VEC_CODO_DER = [VEC_CODO_DER; VectorCodoDer];


    % End efector matrix
    MatrixDer = [MatrixDerRead(i,1) MatrixDerRead(i,2) MatrixDerRead(i,3) MatrixDerRead(i,4);
        MatrixDerRead(i+1,1) MatrixDerRead(i+1,2) MatrixDerRead(i+1,3) MatrixDerRead(i+1,4);
        MatrixDerRead(i+2,1) MatrixDerRead(i+2,2) MatrixDerRead(i+2,3) MatrixDerRead(i+2,4);
        MatrixDerRead(i+3,1) MatrixDerRead(i+3,2) MatrixDerRead(i+3,3) MatrixDerRead(i+3,4)];
    MatrixDer2 = inv(T) * MatrixDer;
    MatrixDer(1:3,1:3)  = rotz(180) * MatrixDer(1:3,1:3); 
    MatrixDer(1:2,4) = -MatrixDer(1:2,4);
    MatrixDer = T * MatrixDer; 

    matrix = MatrixDer; % Para IK_UR

    X_RA_Goal_Der = [MatrixDer2(1,4) MatrixDer2(2,4) MatrixDer2(3,4)]; 
    Goal_Der = [Goal_Der; [MatrixDerRead(i,4) MatrixDerRead(i+1,4) MatrixDerRead(i+2,4)]];
    Goal2_Der = [Goal2_Der;X_RA_Goal_Der]; 
    RotMatDer = [MatrixDer2(1,1) MatrixDer2(1,2) MatrixDer2(1,3); 
        MatrixDer2(2,1) MatrixDer2(2,2) MatrixDer2(2,3);
        MatrixDer2(3,1) MatrixDer2(3,2) MatrixDer2(3,3)];
    EulerAnglesDer = rotm2eul(RotMatDer); 
    Quat_h_der = eul2quat(EulerAnglesDer);
    
    disp("------------- NEW RIGHT ARM ITERATION -------------")
    disp(k)
   
    % IK using UR3_Inverse_Kinematics
    try
        IK_UR % Con MatrixDer

        for ii = 1:1:8
            for jj = 1:1:6 
                configSol(jj).JointPosition = res(ii,jj); 
            end

            % Check collision using the collision model
            setFixedTransform(robotModel.Bodies{1,1}.Joint,T2);

            GoalConfigDer = [configSol(1).JointPosition configSol(2).JointPosition configSol(3).JointPosition configSol(4).JointPosition configSol(5).JointPosition configSol(6).JointPosition];
            [validState,~] = checkCollision(robotModel,GoalConfigDer',env,"IgnoreSelfCollision","off","Exhaustive","on","SkippedSelfCollisions","parent");
            if ~any(validState)

                % End efector for the specific configuration
                EfectorFinalDer = getTransform(robot,configSol,'tool0','base_link'); 
                X_RA_Der = [EfectorFinalDer(1,4) EfectorFinalDer(2,4) EfectorFinalDer(3,4)]; 
                Rot_EF_Der =[EfectorFinalDer(1,1) EfectorFinalDer(1,2) EfectorFinalDer(1,3); 
                    EfectorFinalDer(2,1) EfectorFinalDer(2,2) EfectorFinalDer(2,3);
                    EfectorFinalDer(3,1) EfectorFinalDer(3,2) EfectorFinalDer(3,3)];
                EulerAnglesEFDer = rotm2eul(Rot_EF_Der); 
                Quat_r_der = eul2quat(EulerAnglesEFDer);


                % Wrist 1
                Wrist1Der = getTransform(robot,configSol,'wrist_1_link','base_link');
                S_wrist1_der = [Wrist1Der(1,4) Wrist1Der(2,4) Wrist1Der(3,4)]; % Guada las coordenadas de la muñeca


                % Wrist 2
                Wrist2Der = getTransform(robot,configSol,'wrist_2_link','base_link');
                S_wrist2_der = [Wrist2Der(1,4) Wrist2Der(2,4) Wrist2Der(3,4)];

                % Wrist 3
                Wrist3Der = getTransform(robot,configSol,'wrist_3_link','base_link');
                S_wrist3_der = [Wrist3Der(1,4) Wrist3Der(2,4) Wrist3Der(3,4)];

                %Elbow
                CodoDer = getTransform(robot,configSol,'forearm_link','base_link');
                S_codo_der = [CodoDer(1,4) CodoDer(2,4) CodoDer(3,4)];

                %Shoulder
                HombroSalienteDer = getTransform(robot,configSol,'shoulder_link','base_link');
                S_hombro_der = [HombroSalienteDer(1,4) HombroSalienteDer(2,4) HombroSalienteDer(3,4)];

           
                d_RAx_der = distPosition(X_RA_Der,X_RA_Goal_Der);
                d_RAo_der = distOrientation(Quat_r_der,Quat_h_der);
        
                MDistanciaDer = distanceMetric(VectorCodoDer, S_wrist2_der, S_wrist1_der, S_wrist3_der, S_codo_der, S_hombro_der, Goal_Der');
               
                % Wrist error estimation
                if i == 1 || ~exist('WristOldDer','var')
                    ErroWristDer = 0; 
                else
                    ErroWristDer = variationWrist(configSol(4).JointPosition,WristOldDer); % Calcula el error como la diferencia entre el nuevo valor de la muñeca y el antiguo
                end
                DistFinalDer = real(W_rax*d_RAx_der + W_rao*d_RAo_der + W_A*MDistanciaDer + ErroWrist);
                
                CodoRobotRotIzq = rotx(45)*S_codo_der';
                WristRobotRotIzq= rotx(45)*S_wrist1_der';

                % Check if it is finished
                if d_RAx_der <= 0.05 && d_RAo_der <= 0.16 && ~(CodoRobotRotIzq(2)<-0.1 && WristRobotRotIzq(2)>CodoRobotRotIzq(2)) && DistFinalDer < DistDif && configSol(1).JointPosition >= HombroLim(1) && configSol(1).JointPosition <= HombroLim(2) && configSol(2).JointPosition >= HombroLim2(1) && configSol(2).JointPosition <= HombroLim2(2)
                    DistDif = DistFinalDer;
                    MejorConfig = [configSol(1).JointPosition configSol(2).JointPosition configSol(3).JointPosition configSol(4).JointPosition configSol(5).JointPosition configSol(6).JointPosition];
                    mejor_ii = ii;
                end
            end
        end
        if DistDif ~= DistDigOG
            WristOldIzq = MejorConfig(1,4);
            MEJORES_IZQ = [MEJORES_IZQ;mejor_ii];
            ConfigFinalIzq = [ConfigFinalIzq;MejorConfig];
            ROTMAT_IZQ = [ROTMAT_IZQ;RotMatDer];

        else
            disp('Not valid configuration found')
            WorstIzq = k;
            PEORES_IZQ = [PEORES_IZQ;WorstIzq];
            dif
        end

    catch
        disp('Wrong IK value')
    end   
    
end

% Filter the incorrect data and correct the position values
Goal_Der(PEORES_IZQ,:) = [];
Goal2_Der(PEORES_IZQ,:) = [];
Goal_Der =[Goal_Der(:,1)+0.07,Goal_Der(:,2)+0.13,Goal_Der(:,3)+1.15];


%% Plot the movements of the robotics arm respect the human elbow and wrist

robot = loadrobot("universalUR3");
T1 = [1 0 0 0.07;0 0.7071 0.7071 0.13;0 -0.7071 0.7071 1.15;0 0 0 1];
setFixedTransform(robot.Bodies{1,1}.Joint,T1);
configuraciones = robot.homeConfiguration;
k=0;
DISTANCIAS = [];
END = [];
ROTEND = [];
X =[];
Y = [];
Z = [];

for i=1:1:length(ConfigFinalIzq)
    k= k+1
    configuraciones(1).JointPosition = ConfigFinalIzq(i,1);
    configuraciones(2).JointPosition = ConfigFinalIzq(i,2);
    configuraciones(3).JointPosition = ConfigFinalIzq(i,3);
    configuraciones(4).JointPosition = ConfigFinalIzq(i,4);
    configuraciones(5).JointPosition = ConfigFinalIzq(i,5);
    configuraciones(6).JointPosition = ConfigFinalIzq(i,6);
    zoom(gca,'on');
    show(robot,configuraciones,"Collisions","off");
    CodoR = getTransform(robot,configuraciones,'forearm_link','base_link');
    CODRob_IZQ = [CODRob_IZQ;CodoR(1,4) CodoR(2,4) CodoR(3,4)];
    camzoom(1.7);
    hold on
    show(env1{1});
    show(env1{2});
    show(env1{3});
    show(env1{4});
    hold on
    plot3(VEC_CODO_DER(k,1)+0.07,VEC_CODO_DER(k,2)+0.13,VEC_CODO_DER(k,3)+1.15,'o','Color','g','MarkerSize',10,'MarkerFaceColor','g')
    plot3(Goal_Der(k,1),Goal_Der(k,2),Goal_Der(k,3),'o','Color','r','MarkerSize',10,'MarkerFaceColor','r')
    hold on
    PuntoEnd = getTransform(robot,configuraciones,'tool0','base_link');
    Punto = [Goal_Der(k,1),Goal_Der(k,2),Goal_Der(k,3)];
    END = [END;PuntoEnd(1,4),PuntoEnd(2,4),PuntoEnd(3,4)];
    ROTEND = [ROTEND;PuntoEnd(1,1),PuntoEnd(1,2),PuntoEnd(1,3);PuntoEnd(2,1),PuntoEnd(2,2),PuntoEnd(2,3);PuntoEnd(3,1),PuntoEnd(3,2),PuntoEnd(3,3)];

    pause(0.001);
    hold off
end


%% Plot in 3D (smoothed to visualized it better)

figure;
plot3(Goal2_Der(:,1),Goal2_Der(:,2),Goal2_Der(:,3),'Color','g');
hold on
xlabel('X (m)')
ylabel('Y (m)')
zlabel('Z (m)')
plot3(END(:,1),END(:,2),END(:,3),'Color','b');
plot3(Goal2_Der(1,1),Goal2_Der(1,2),Goal2_Der(1,3),'o','Color','r','MarkerSize',8,'MarkerFaceColor','r');
plot3(Goal2_Der(length(Goal2_Der),1),Goal2_Der(length(Goal2_Der),2),Goal2_Der(length(Goal2_Der),3),'o','Color','m','MarkerSize',8,'MarkerFaceColor','m');
[t,s]=title("Comparision between data acquired");
t.FontSize = 16;
legend('Robot data','Human data','Initial point','Final point')
hold off

%% Plot the error values of the end efector to compare

PlotError