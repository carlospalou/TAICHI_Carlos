%% Human-like code for MATLAB
clear;clc;close all;

%% Load the robot model
robotIzq = loadrobot("universalUR3"); % Usa una funcion de Robotics System Toolbox para cargar el robot UR3
robotDer = loadrobot("universalUR3");
TI = [1 0 0 0;0 0.7071 0.7071 0;0 -0.7071 0.7071 0;0 0 0 1]; % Matriz del brazo izquierdo solo rotacion, sin desplazamiento, rotacion de -45º en X
TD = [1 0 0 0;0 -0.7071 -0.7071 0;0 -0.7071 0.7071 0;0 0 0 1]; % Matriz del brazo izquierdo, rotacion 45º en X y 'flip'
HombroLim = [-3.14,3.14];
HombroLim2 = [-2, 1.5];
weights = [2, 3.42, 1.26, 0.8, 0.8, 0.35];
configSolnIzq = robotIzq.homeConfiguration; % La configuracion predeterminada del robot
configSolnDer = robotDer.homeConfiguration;

%% Collision environment 
robotModelIzq = loadrobot("universalUR3","DataFormat","column"); % Para utilizar vectores columna en las configuraciones de articulacion
TIzq = [1 0 0 0.07;0 0.7071 0.7071 0.13;0 -0.7071 0.7071 1.15;0 0 0 1]; % Matriz brazo izquierdo, rotacion y desplazamiento
setFixedTransform(robotModelIzq.Bodies{1,1}.Joint,TIzq);
ssI = manipulatorStateSpace(robotModelIzq); % Crea un espacio de estados para representar y manipular el espacio de configuraciones del robot, usa Navigation Toolbox
svI = manipulatorCollisionBodyValidator(ssI, SkippedSelfCollisions="parent"); % Validador de colisiones para el espacio de estados
svI.ValidationDistance = 0.1; 
svI.IgnoreSelfCollision = true; % Ignora el propio brazo como colision 

robotModelDer = loadrobot("universalUR3","DataFormat","column"); 
TDer = [1 0 0 0.07;0 -0.7071 -0.7071 -0.13;0 -0.7071 0.7071 1.15;0 0 0 1];
setFixedTransform(robotModelDer.Bodies{1,1}.Joint,TDer);
ssD = manipulatorStateSpace(robotModelDer); 
svD = manipulatorCollisionBodyValidator(ssD, SkippedSelfCollisions="parent"); 
svD.ValidationDistance = 0.1; 
svD.IgnoreSelfCollision = true;  

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

env = {body1 body2 body3 body4 }; % Crea un entorno
svI.Environment = env;
svD.Environment = env;

% To visualize the environment
figure(1)
show(robotModelIzq,"Collisions","off");
hold on
show(robotModelDer,"Collisions","off");
show(env{1});
show(env{2});
show(env{3});
show(env{4});
hold off


%% Inverse kinematics solver

% Read the files
path_izq = '/home/carlos/TAICHI_Carlos/HumanData/Prueba1/DatosBrazoIzquierdo.csv';
path2_izq = '/home/carlos/TAICHI_Carlos/HumanData/Prueba1/CodoIzquierdo.csv';
path_der = '/home/carlos/TAICHI_Carlos/HumanData/Prueba1/DatosBrazoDerecho.csv';
path2_der = '/home/carlos/TAICHI_Carlos/HumanData/Prueba1/CodoDerecho.csv';

MatrixIzqRead = readmatrix(path_izq); % Almacena los datos del .csv en una matriz
CodoIzqRead = readmatrix(path2_izq);
MatrixDerRead = readmatrix(path_der);
CodoDerRead = readmatrix(path2_der);

iter_izq = length(MatrixIzqRead); % Cantidad de elementos de la matriz
iter2_izq = length(CodoIzqRead);
iter_der = length(MatrixDerRead);
iter2_der = length(CodoDerRead);

CodoIzqOrganizado = []; % Matriz con los vectores de posicion del codo, tres columnas y tantas filas como vectores
CodoDerOrganizado = [];

VEC_CODO_IZQ = []; % Matriz de vectores de posicion del codo, tres columnas y tantas filas como iteraciones del primer bucle for
VEC_CODO_DER = [];

MEJORES_IZQ = []; % Guarda los indices de las mejores configuraciones
MEJORES_DER = [];
ConfigFinalIzq= []; % Guarda la posiocion de las articulaciones de las mejores configuraciones
ConfigFinalDer= [];
ROTMAT_IZQ = []; % Guarda la matriz de rotacion objetivo
ROTMAT_DER = [];

PEORES_IZQ = []; % Guarda las iteraciones en las que no se ha conseguido una configuracion valida
PEORES_DER = [];

RES=[]; % Resultados de IK_UR.m

% Organised the elbow values
for j=1:3:iter2_izq  
    CodoIzqOrganizado = [CodoIzqOrganizado; CodoIzqRead(j,1) CodoIzqRead(j+1) CodoIzqRead(j+2)]; % Vuelve a poner los datos del codo en tres columnas ya que en el .csv estaban en una unica columna
end

for j=1:3:iter2_der  
    CodoDerOrganizado = [CodoDerOrganizado; CodoDerRead(j,1) CodoDerRead(j+1) CodoDerRead(j+2)];
end


%% Main 

% Weigths values
WX= 1; % Position weigth
WO = 10; % Orientation weigth
WH = 50; % Humanity weigth

% Evaluation loop
Goal_Izq = []; % Goal point with respect to human shoulder 
Goal2_Izq = []; % Goal point with respect to human shoulder rotated 45
MatrixCodoIzq = zeros(4,4);
Goal_Der = [];
Goal2_Der = [];
MatrixCodoDer = zeros(4,4);

k=0;

for i=1:4:iter_izq % i va de 4 en 4 porque las matrices de transformacion homogeneas son 4x4
    k = k+1; % Numero de iteraciones del bucle
    DistMinIzq = 10000000;
    Dist = DistMinIzq;
    iteration = 0;
    check = true;
    
    % Elbow matrix
    VectorCodoIzq = [CodoIzqOrganizado(k,1), CodoIzqOrganizado(k,2), CodoIzqOrganizado(k,3)]; % Coge uno a uno los vectores de posicion del codo
    VEC_CODO_IZQ = [VEC_CODO_IZQ; VectorCodoIzq]; % Va almacenando los vectores anteriores en una matriz, supongo que esto se hace para que haya el mismo numero de vectores de posicion del codo que de matrices de transformacion homogeneas

    % End Efector matrix
    MatrixIzq = [MatrixIzqRead(i,1) MatrixIzqRead(i,2) MatrixIzqRead(i,3) MatrixIzqRead(i,4); % Matriz de transformacion homogenea del EF obtenida de los datos de brazoProf2.py
        MatrixIzqRead(i+1,1) MatrixIzqRead(i+1,2) MatrixIzqRead(i+1,3) MatrixIzqRead(i+1,4);
        MatrixIzqRead(i+2,1) MatrixIzqRead(i+2,2) MatrixIzqRead(i+2,3) MatrixIzqRead(i+2,4);
        MatrixIzqRead(i+3,1) MatrixIzqRead(i+3,2) MatrixIzqRead(i+3,3) MatrixIzqRead(i+3,4)];
    
    MatrixIzq2 = inv(TI) * MatrixIzq; % Rotacion de 45º en X, rotacion contraria de TI
    
    MatrixIzq(1:3,1:3)  = rotz(180) * MatrixIzq(1:3,1:3); % De la matriz de transformacion homogenea del EF se coge solo la rotacion y se le aplica una rotacion de 180º en Z
    MatrixIzq(1:2,4) = -MatrixIzq(1:2,4); % Se invierten los valores de X e Y en la matriz de transformacion homogenea
    MatrixIzq = TI * MatrixIzq; % Rotacion de -45º en X
    matrix = MatrixIzq; % Para IK_UR

    Goal_Izq = [Goal_Izq; [MatrixIzqRead(i,4) MatrixIzqRead(i+1,4) MatrixIzqRead(i+2,4)]]; % Posiciones del EF obtenidas directamente de brazoProf2.py
    
    X_Goal2_Izq = [MatrixIzq2(1,4) MatrixIzq2(2,4) MatrixIzq2(3,4)]; % Posicion del EF aplicada la rotacion de 45º en X
    Goal2_Izq = [Goal2_Izq;X_Goal2_Izq]; % Posiciones anteriores almacenadas en una matriz 
    Rot_Goal2_Izq = [MatrixIzq2(1,1) MatrixIzq2(1,2) MatrixIzq2(1,3); % Coge solo las rotaciones
        MatrixIzq2(2,1) MatrixIzq2(2,2) MatrixIzq2(2,3);
        MatrixIzq2(3,1) MatrixIzq2(3,2) MatrixIzq2(3,3)];
    EulerAngles_Goal2_Izq = rotm2eul(Rot_Goal2_Izq); % Pasa de matriz de rotacion a angulos de Euler
    Quat_Goal2_Izq = eul2quat(EulerAngles_Goal2_Izq); % Pasa de angulos de Euler a cuaterniones
    
    disp("------------- NEW LEFT ARM ITERATION -------------")
    disp(k)
   
    % IK using UR3_Inverse_Kinematics
    try
        IK_UR % Con MatrixIzq

        for ii = 1:1:8
            for jj = 1:1:6 % res son los resultados de UR3_Inverse_Kinematics, 8 soluciones con 6 articulaciones
                configSolnIzq(jj).JointPosition = res(ii,jj); % Asigna la posicion de las articulaciones para cada configuracion
            end

            % Check collision using the collision model
            GoalConfigIzq = [configSolnIzq(1).JointPosition configSolnIzq(2).JointPosition configSolnIzq(3).JointPosition configSolnIzq(4).JointPosition configSolnIzq(5).JointPosition configSolnIzq(6).JointPosition];
            [validState,~] = checkCollision(robotModelIzq,GoalConfigIzq',env,"IgnoreSelfCollision","off","Exhaustive","on","SkippedSelfCollisions","parent");
            if ~any(validState)

                % End efector for the specific configuration
                EfectorFinalIzq = getTransform(robotIzq,configSolnIzq,'tool0','base_link'); % Matriz de transformacion homogenea del EF respecto a la base en funcion de la configuracion
                X_Izq = [EfectorFinalIzq(1,4) EfectorFinalIzq(2,4) EfectorFinalIzq(3,4)]; % Coge solo la posicion
                Rot_Izq =[EfectorFinalIzq(1,1) EfectorFinalIzq(1,2) EfectorFinalIzq(1,3); % Coge solo la rotacion
                    EfectorFinalIzq(2,1) EfectorFinalIzq(2,2) EfectorFinalIzq(2,3);
                    EfectorFinalIzq(3,1) EfectorFinalIzq(3,2) EfectorFinalIzq(3,3)];
                EulerAngles_Izq = rotm2eul(Rot_Izq); 
                Quat_Izq = eul2quat(EulerAngles_Izq);

                % Wrist 1
                Wrist1Izq = getTransform(robotIzq,configSolnIzq,'wrist_1_link','base_link');
                X_Wrist1Izq = [Wrist1Izq(1,4) Wrist1Izq(2,4) Wrist1Izq(3,4)]; % Guada las coordenadas de la muñeca

                % Wrist 2
                Wrist2Izq = getTransform(robotIzq,configSolnIzq,'wrist_2_link','base_link');
                X_Wrist2Izq = [Wrist2Izq(1,4) Wrist2Izq(2,4) Wrist2Izq(3,4)];

                % Wrist 3
                Wrist3Izq = getTransform(robotIzq,configSolnIzq,'wrist_3_link','base_link');
                X_Wrist3Izq = [Wrist3Izq(1,4) Wrist3Izq(2,4) Wrist3Izq(3,4)];

                % Elbow
                CodoIzq = getTransform(robotIzq,configSolnIzq,'forearm_link','base_link');
                X_CodoIzq = [CodoIzq(1,4) CodoIzq(2,4) CodoIzq(3,4)];

                % Shoulder
                HombroSalienteIzq = getTransform(robotIzq,configSolnIzq,'shoulder_link','base_link');
                X_HombroIzq = [HombroSalienteIzq(1,4) HombroSalienteIzq(2,4) HombroSalienteIzq(3,4)];

                % EE error estimation
                DistXIzq = distPosition(X_Izq,X_Goal2_Izq); % Calcula la diferencia entre la posicion del EF obtenida de los datos de la camara y de la solucion de IK_UR
                
                % Orientation error estimation
                DistOIzq= distOrientation(Quat_Izq,Quat_Goal2_Izq); % Calcula la diferencia de orientacion del EF
                
                % Elbow error estimation
                DistMIzq = distanceMetric(VectorCodoIzq, X_Wrist2Izq, X_Wrist1Izq, X_Wrist3Izq, X_CodoIzq, X_HombroIzq, Goal_Izq'); 
               
                % Wrist error estimation
                if i == 1 || ~exist('WristOldIzq','var')
                    ErroWristIzq = 0; % Si es la primera iteracion o no hay un valor para la mejor configuracion el error es 0
                else
                    ErroWristIzq = variationWrist(configSolnIzq(4).JointPosition,WristOldIzq); % Calcula el error como la diferencia entre el nuevo valor de la muñeca y el antiguo
                end
                
                % Final error
                DistIzq = real(WX*DistXIzq + WO*DistOIzq + WH*DistMIzq + ErroWristIzq); % Pondera los errores calculados anteriormente
                
                CodoRotIzq = rotx(45)*X_CodoIzq';
                WristRotIzq= rotx(45)*X_Wrist1Izq';

                % Check if it is finished
                if DistXIzq <= 0.05 && DistOIzq <= 0.16 && ~(CodoRotIzq(2) < -0.1 && WristRotIzq(2) > CodoRotIzq(2)) && DistIzq < DistMinIzq && configSolnIzq(1).JointPosition >= HombroLim(1) && configSolnIzq(1).JointPosition <= HombroLim(2) && configSolnIzq(2).JointPosition >= HombroLim2(1) && configSolnIzq(2).JointPosition <= HombroLim2(2)
                    DistMinIzq = DistIzq; % Si se cumplen los limites, que incluyen que la distancia sea menor, pasa a ser la distancia minima
                    MejorConfigIzq = [configSolnIzq(1).JointPosition configSolnIzq(2).JointPosition configSolnIzq(3).JointPosition configSolnIzq(4).JointPosition configSolnIzq(5).JointPosition configSolnIzq(6).JointPosition];
                    mejor_ii = ii; % Se guarda la mejor configuracion
                end
            end
        end

        if DistMinIzq ~= Dist % Si la distancia minima no es la distancia inicial
            WristOldIzq = MejorConfigIzq(1,4); 
            MEJORES_IZQ = [MEJORES_IZQ;mejor_ii];
            ConfigFinalIzq = [ConfigFinalIzq; MejorConfigIzq];
            ROTMAT_IZQ = [ROTMAT_IZQ;Rot_Goal2_Izq];

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
    DistMinDer = 10000000;
    Dist = DistMinDer;
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

    MatrixDer2 = inv(TI) * MatrixDer;

    MatrixDer(1:3,1:3)  = rotz(180) * MatrixDer(1:3,1:3); 
    MatrixDer(1:2,4) = -MatrixDer(1:2,4);
    MatrixDer = TI * MatrixDer; 
    matrix = MatrixDer; % Para IK_UR

    Goal_Der = [Goal_Der; [MatrixDerRead(i,4) MatrixDerRead(i+1,4) MatrixDerRead(i+2,4)]];

    X_Goal2_Der = [MatrixDer2(1,4) MatrixDer2(2,4) MatrixDer2(3,4)]; 
    Goal2_Der = [Goal2_Der;X_Goal2_Der]; 
    Rot_Goal2_Der = [MatrixDer2(1,1) MatrixDer2(1,2) MatrixDer2(1,3); 
        MatrixDer2(2,1) MatrixDer2(2,2) MatrixDer2(2,3);
        MatrixDer2(3,1) MatrixDer2(3,2) MatrixDer2(3,3)];
    EulerAngles_Goal2_Der = rotm2eul(Rot_Goal2_Der); 
    Quat_Goal2_Der = eul2quat(EulerAngles_Goal2_Der);
    
    disp("------------- NEW RIGHT ARM ITERATION -------------")
    disp(k)
   
    try
        IK_UR % Con MatrixDer

        for ii = 1:1:8
            for jj = 1:1:6 
                configSolnDer(jj).JointPosition = res(ii,jj); 
            end

            GoalConfigDer = [configSolnDer(1).JointPosition configSolnDer(2).JointPosition configSolnDer(3).JointPosition configSolnDer(4).JointPosition configSolnDer(5).JointPosition configSolnDer(6).JointPosition];
            [validState,~] = checkCollision(robotModelDer,GoalConfigDer',env,"IgnoreSelfCollision","off","Exhaustive","on","SkippedSelfCollisions","parent");
            if ~any(validState)

                % End efector for the specific configuration
                EfectorFinalDer = getTransform(robotIzq,configSolnDer,'tool0','base_link'); 
                X_Der = [EfectorFinalDer(1,4) EfectorFinalDer(2,4) EfectorFinalDer(3,4)]; 
                Rot_Der =[EfectorFinalDer(1,1) EfectorFinalDer(1,2) EfectorFinalDer(1,3); 
                    EfectorFinalDer(2,1) EfectorFinalDer(2,2) EfectorFinalDer(2,3);
                    EfectorFinalDer(3,1) EfectorFinalDer(3,2) EfectorFinalDer(3,3)];
                EulerAngles_Der = rotm2eul(Rot_Der); 
                Quat_Der = eul2quat(EulerAngles_Der);

                % Wrist 1
                Wrist1Der = getTransform(robotIzq,configSolnDer,'wrist_1_link','base_link');
                X_Wrist1Der = [Wrist1Der(1,4) Wrist1Der(2,4) Wrist1Der(3,4)]; % Guada las coordenadas de la muñeca

                % Wrist 2
                Wrist2Der = getTransform(robotIzq,configSolnDer,'wrist_2_link','base_link');
                X_Wrist2Der = [Wrist2Der(1,4) Wrist2Der(2,4) Wrist2Der(3,4)];

                % Wrist 3
                Wrist3Der = getTransform(robotIzq,configSolnDer,'wrist_3_link','base_link');
                X_Wrist3Der = [Wrist3Der(1,4) Wrist3Der(2,4) Wrist3Der(3,4)];

                % Elbow
                CodoDer = getTransform(robotIzq,configSolnDer,'forearm_link','base_link');
                X_CodoDer = [CodoDer(1,4) CodoDer(2,4) CodoDer(3,4)];

                % Shoulder
                HombroSalienteDer = getTransform(robotIzq,configSolnDer,'shoulder_link','base_link');
                X_HombroDer = [HombroSalienteDer(1,4) HombroSalienteDer(2,4) HombroSalienteDer(3,4)];
           
                % Error estimation
                DistXDer = distPosition(X_Der,X_Goal2_Der);
                DistODer = distOrientation(Quat_Der,Quat_Goal2_Der);
                DistMDer = distanceMetric(VectorCodoDer, X_Wrist2Der, X_Wrist1Der, X_Wrist3Der, X_CodoDer, X_HombroDer, Goal_Der');
               
                % Wrist error estimation
                if i == 1 || ~exist('WristOldDer','var')
                    ErroWristDer = 0; 
                else
                    ErroWristDer = variationWrist(configSolnDer(4).JointPosition,WristOldDer); % Calcula el error como la diferencia entre el nuevo valor de la muñeca y el antiguo
                end

                DistDer = real(WX*DistXDer + WO*DistODer + WH*DistMDer + ErroWristDer);
                
                CodoRobotRotDer = rotx(45)*X_CodoDer';
                WristRobotRotDer= rotx(45)*X_Wrist1Der';

                % Check if it is finished
                if DistXDer <= 0.05 && DistODer <= 0.16 && ~(CodoRobotRotDer(2)<-0.1 && WristRobotRotDer(2)>CodoRobotRotDer(2)) && DistDer < DistMinDer && configSolnDer(1).JointPosition >= HombroLim(1) && configSolnDer(1).JointPosition <= HombroLim(2) && configSolnDer(2).JointPosition >= HombroLim2(1) && configSolnDer(2).JointPosition <= HombroLim2(2)
                    DistMinDer = DistDer;
                    MejorConfigDer = [configSolnDer(1).JointPosition configSolnDer(2).JointPosition configSolnDer(3).JointPosition configSolnDer(4).JointPosition configSolnDer(5).JointPosition configSolnDer(6).JointPosition];
                    mejor_ii = ii;
                end
            end
        end

        if DistMinDer ~= Dist
            WristOldDer = MejorConfigDer(1,4);
            MEJORES_DER = [MEJORES_DER;mejor_ii];
            ConfigFinalDer = [ConfigFinalDer;MejorConfigDer];
            ROTMAT_DER = [ROTMAT_DER;Rot_Goal2_Der];

        else
            disp('Not valid configuration found')
            WorstDer = k;
            PEORES_DER = [PEORES_DER;WorstDer];
            dif
        end

    catch
        disp('Wrong IK value')
    end   
    
end

% Filter the incorrect data and correct the position values
Goal_Izq(PEORES_IZQ,:) = [];
Goal2_Izq(PEORES_IZQ,:) = [];
Goal_Izq =[Goal_Izq(:,1)+0.07,Goal_Izq(:,2)+0.13,Goal_Izq(:,3)+1.15];

Goal_Der(PEORES_DER,:) = [];
Goal2_Der(PEORES_DER,:) = [];
Goal_Der =[Goal_Der(:,1)+0.07,Goal_Der(:,2)+0.13,Goal_Der(:,3)+1.15];


%% Plot the movements of the robotics arm respect the human elbow and wrist

robotIzq = loadrobot("universalUR3");
setFixedTransform(robotIzq.Bodies{1,1}.Joint,TIzq);
configuracionesIzq = robotIzq.homeConfiguration;

robotDer = loadrobot("universalUR3");
setFixedTransform(robotDer.Bodies{1,1}.Joint,TDer);
configuracionesDer = robotDer.homeConfiguration;

COD_IZQ = []; 
COD_DER = [];
DISTANCIAS_IZQ = [];
DISTANCIAS_DER = [];
END_IZQ = [];
END_DER = [];
ROTEND_IZQ = [];
ROTEND_DER = [];
X_IZQ = [];
X_DER = [];
Y_IZQ = [];
Y_DER = [];
Z_IZQ = [];
Z_DER = [];

k=0;

for i=1:1:length(ConfigFinalIzq)
    k = k+1;

    configuracionesIzq(1).JointPosition = ConfigFinalIzq(i,1); % Establece la ConfigFinal como la configuracion del robot
    configuracionesIzq(2).JointPosition = ConfigFinalIzq(i,2);
    configuracionesIzq(3).JointPosition = ConfigFinalIzq(i,3);
    configuracionesIzq(4).JointPosition = ConfigFinalIzq(i,4);
    configuracionesIzq(5).JointPosition = ConfigFinalIzq(i,5);
    configuracionesIzq(6).JointPosition = ConfigFinalIzq(i,6);
    
    figure(2);
    %zoom(gca,'on');
    show(robotIzq,configuracionesIzq,"Collisions","off"); % Muestra el robot con la ConfigFinal
    camzoom(1.7); 
    hold on
    show(env{1});
    show(env{2});
    show(env{3});
    show(env{4}); % Muestra el cuerpo del robot
    hold off

    Codo_Izq = getTransform(robotIzq,configuracionesIzq,'forearm_link','base_link'); 
    COD_IZQ = [COD_IZQ; Codo_Izq(1,4) Codo_Izq(2,4) Codo_Izq(3,4)]; % Guarda las posiciones del codo respecto a la base, el hombro

    figure(3);
    plot3(VEC_CODO_IZQ(k,1)+0.07, VEC_CODO_IZQ(k,2)+0.13, VEC_CODO_IZQ(k,3)+1.15,'o','Color','g','MarkerSize',10,'MarkerFaceColor','g')
    plot3(Goal_Izq(k,1),Goal_Izq(k,2),Goal_Izq(k,3),'o','Color','r','MarkerSize',10,'MarkerFaceColor','r')

    PuntoEndIzq = getTransform(robotIzq,configuracionesIzq,'tool0','base_link');
    PuntoIzq = [Goal_Izq(k,1),Goal_Izq(k,2),Goal_Izq(k,3)];
    END_IZQ = [END_IZQ; PuntoEndIzq(1,4), PuntoEndIzq(2,4), PuntoEndIzq(3,4)];
    ROTEND_IZQ = [ROTEND_IZQ; PuntoEndIzq(1,1),PuntoEndIzq(1,2),PuntoEndIzq(1,3); PuntoEndIzq(2,1),PuntoEndIzq(2,2),PuntoEndIzq(2,3); PuntoEndIzq(3,1),PuntoEndIzq(3,2),PuntoEndIzq(3,3)];

end

k=0;

for i=1:1:length(ConfigFinalDer)

    k = k+1
    configuracionesIzq(1).JointPosition = ConfigFinalDer(i,1);
    configuracionesIzq(2).JointPosition = ConfigFinalDer(i,2);
    configuracionesIzq(3).JointPosition = ConfigFinalDer(i,3);
    configuracionesIzq(4).JointPosition = ConfigFinalDer(i,4);
    configuracionesIzq(5).JointPosition = ConfigFinalDer(i,5);
    configuracionesIzq(6).JointPosition = ConfigFinalDer(i,6);
    zoom(gca,'on');
    show(robotIzq,configuracionesIzq,"Collisions","off");
    Codo_Der = getTransform(robotIzq,configuracionesIzq,'forearm_link','base_link');
    COD_DER = [COD_DER; Codo_Der(1,4) Codo_Der(2,4) Codo_Der(3,4)];
    camzoom(1.7);
    hold on

    show(env{1});
    show(env{2});
    show(env{3});
    show(env{4});
    hold on

    plot3(VEC_CODO_DER(k,1)+0.07,VEC_CODO_DER(k,2)+0.13,VEC_CODO_DER(k,3)+1.15,'o','Color','g','MarkerSize',10,'MarkerFaceColor','g')
    plot3(Goal_Der(k,1),Goal_Der(k,2),Goal_Der(k,3),'o','Color','r','MarkerSize',10,'MarkerFaceColor','r')
    hold on

    PuntoEndDer = getTransform(robotIzq,configuracionesIzq,'tool0','base_link');
    PuntoDer = [Goal_Der(k,1),Goal_Der(k,2),Goal_Der(k,3)];
    END_DER = [END_DER;PuntoEndDer(1,4),PuntoEndDer(2,4),PuntoEndDer(3,4)];
    ROTEND_DER = [ROTEND_DER;PuntoEndDer(1,1),PuntoEndDer(1,2),PuntoEndDer(1,3);PuntoEndDer(2,1),PuntoEndDer(2,2),PuntoEndDer(2,3);PuntoEndDer(3,1),PuntoEndDer(3,2),PuntoEndDer(3,3)];

    pause(0.001);
    hold off
end



%% Plot in 3D (smoothed to visualized it better)

figure;
plot3(Goal2_Izq(:,1),Goal2_Izq(:,2),Goal2_Izq(:,3),'Color','g');
hold on
xlabel('X (m)')
ylabel('Y (m)')
zlabel('Z (m)')
plot3(END_IZQ(:,1),END_IZQ(:,2),END_IZQ(:,3),'Color','b');
plot3(Goal2_Izq(1,1),Goal2_Izq(1,2),Goal2_Izq(1,3),'o','Color','r','MarkerSize',8,'MarkerFaceColor','r');
plot3(Goal2_Izq(length(Goal2_Izq),1),Goal2_Izq(length(Goal2_Izq),2),Goal2_Izq(length(Goal2_Izq),3),'o','Color','m','MarkerSize',8,'MarkerFaceColor','m');
[t,s]=title("Comparision between data acquired");
t.FontSize = 16;
legend('Robot data','Human data','Initial point','Final point')
hold off

figure;
plot3(Goal2_Der(:,1),Goal2_Der(:,2),Goal2_Der(:,3),'Color','g');
hold on
xlabel('X (m)')
ylabel('Y (m)')
zlabel('Z (m)')
plot3(END_DER(:,1),END_DER(:,2),END_DER(:,3),'Color','b');
plot3(Goal2_Der(1,1),Goal2_Der(1,2),Goal2_Der(1,3),'o','Color','r','MarkerSize',8,'MarkerFaceColor','r');
plot3(Goal2_Der(length(Goal2_Der),1),Goal2_Der(length(Goal2_Der),2),Goal2_Der(length(Goal2_Der),3),'o','Color','m','MarkerSize',8,'MarkerFaceColor','m');
[t,s]=title("Comparision between data acquired");
t.FontSize = 16;
legend('Robot data','Human data','Initial point','Final point')
hold off

%% Plot the error values of the end efector to compare
