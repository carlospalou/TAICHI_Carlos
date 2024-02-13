%% Human-like code for MATLAB
clear;clc;close all;

%% Load the robot model
robotIzq = loadrobot("universalUR3"); % Usa una funcion de Robotics System Toolbox para cargar el robot UR3
robotDer = loadrobot("universalUR3");
TI = [1 0 0 0;0 0.7071 0.7071 0;0 -0.7071 0.7071 0;0 0 0 1]; % Matriz del brazo izquierdo solo rotacion, sin desplazamiento, rotacion de -45º en X
TD = [1 0 0 0;0 0.7071 -0.7071 0;0 0.7071 0.7071 0;0 0 0 1]; % Matriz del brazo izquierdo, rotacion 45º en X 
HombroLimIzq = [-3.14, 3.14];
HombroLim2Izq = [-2, 1.5];
HombroLimDer = [0, 2*3.14];
HombroLim2Der = [-2+3.14, 1.5+3.14];
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
TDer = [1 0 0 0.07;0 0.7071 -0.7071 -0.13;0 0.7071 0.7071 1.15;0 0 0 1];
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

%To visualize the environment
% figure(1)
% show(robotModelIzq,"Collisions","off");
% hold on
% show(robotModelDer,"Collisions","off");
% show(env{1});
% show(env{2});
% show(env{3});
% show(env{4});
% hold off
% pause();

%% Inverse kinematics solver

% Read the files
path_izq = '/home/carlos/TAICHI_Carlos/HumanData/Prueba32/DatosBrazoIzquierdo.csv';
path2_izq = '/home/carlos/TAICHI_Carlos/HumanData/Prueba32/CodoIzquierdo.csv';
path_der = '/home/carlos/TAICHI_Carlos/HumanData/Prueba32/DatosBrazoDerecho.csv';
path2_der = '/home/carlos/TAICHI_Carlos/HumanData/Prueba32/CodoDerecho.csv';

MatrixIzqRead = readmatrix(path_izq); % Almacena los datos del .csv en una matriz
CodoIzqRead = readmatrix(path2_izq);
MatrixDerRead = readmatrix(path_der);
CodoDerRead = readmatrix(path2_der);

iter_izq = length(MatrixIzqRead); % Cantidad de elementos de la matriz
iter2_izq = length(CodoIzqRead);
iter_der = length(MatrixDerRead);
iter2_der = length(CodoDerRead);

CodoIzqOrganizado = []; % Matriz con las posiciones del codo del humano, tres columnas y tantas filas como vectores
CodoDerOrganizado = [];

VEC_CODO_IZQ = []; % Matriz de vectores de posicion del codo del humano, tres columnas y tantas filas como iteraciones del primer bucle for
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

ki=0;

for i=1:4:iter_izq % i va de 4 en 4 porque las matrices de transformacion homogeneas son 4x4
    ki = ki+1; % Numero de iteraciones del bucle
    DistMinIzq = 10000000;
    DistI = DistMinIzq;
    iteration = 0;
    check = true;
    
    % Elbow matrix
    VectorCodoIzq = [CodoIzqOrganizado(ki,1), CodoIzqOrganizado(ki,2), CodoIzqOrganizado(ki,3)]; % Coge una fila de la matriz de posiciones de del codo y la guarda como vector
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
    
    disp(' ')
    disp("------------- NEW LEFT ARM ITERATION -------------")
    disp(ki)
   
    % IK using UR3_Inverse_Kinematics
    try
        IK_UR % Con MatrixIzq
        %disp(res)

        for ii = 1:1:8
            for jj = 1:1:6 % res son los resultados de UR3_Inverse_Kinematics, 8 soluciones con 6 articulaciones
                configSolnIzq(jj).JointPosition = res(ii,jj); % Asigna la posicion de las articulaciones para cada configuracion
            end

            % Check collision using the collision model
            GoalConfigIzq = [configSolnIzq(1).JointPosition configSolnIzq(2).JointPosition configSolnIzq(3).JointPosition configSolnIzq(4).JointPosition configSolnIzq(5).JointPosition configSolnIzq(6).JointPosition];
            [validState,~] = checkCollision(robotModelIzq,GoalConfigIzq',env,"IgnoreSelfCollision","off","Exhaustive","on","SkippedSelfCollisions","parent");
            if ~any(validState)
                %disp('No collisions')

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
                %disp(DistIzq)

                CodoRotIzq = rotx(45)*X_CodoIzq';
                WristRotIzq= rotx(45)*X_Wrist1Izq';

                % Check if it is finished
                if DistXIzq <= 0.05 && DistOIzq <= 0.16 && ~(CodoRotIzq(2) < -0.1 && WristRotIzq(2) > CodoRotIzq(2)) && DistIzq < DistMinIzq && configSolnIzq(1).JointPosition >= HombroLimIzq(1) && configSolnIzq(1).JointPosition <= HombroLimIzq(2) && configSolnIzq(2).JointPosition >= HombroLim2Izq(1) && configSolnIzq(2).JointPosition <= HombroLim2Izq(2)
                    DistMinIzq = DistIzq; % Si se cumplen los limites, que incluyen que la distancia sea menor, pasa a ser la distancia minima
                    MejorConfigIzq = [configSolnIzq(1).JointPosition configSolnIzq(2).JointPosition configSolnIzq(3).JointPosition configSolnIzq(4).JointPosition configSolnIzq(5).JointPosition configSolnIzq(6).JointPosition];
                    mejor_ii = ii; % Se guarda la mejor configuracion
                end

            else
                %disp('Collision')        
            end

        end

        if DistMinIzq ~= DistI % Si la distancia minima no es la distancia inicial
            disp(' ')
            disp('Valid configuration found')
            WristOldIzq = MejorConfigIzq(1,4); 
            MEJORES_IZQ = [MEJORES_IZQ;mejor_ii];
            ConfigFinalIzq = [ConfigFinalIzq; MejorConfigIzq];
            ROTMAT_IZQ = [ROTMAT_IZQ;Rot_Goal2_Izq];

        elseif exist('MejorConfigIzq', 'var')
            disp(' ')
            disp('Not valid configuration found')
            ConfigFinalIzq = [ConfigFinalIzq; MejorConfigIzq];
            WorstIzq = ki;
            PEORES_IZQ = [PEORES_IZQ;WorstIzq];

        else 
            disp(' ')
            disp('Not any valid configuration found')
            WorstIzq = ki;
            PEORES_IZQ = [PEORES_IZQ;WorstIzq];
        end

    catch
        disp(' ')
        disp('Wrong IK value')
    end   
    
end


kd=0;

% PARTE DERECHA

for i=1:4:iter_der
    kd = kd+1;
    DistMinDer = 10000000;
    DistR = DistMinDer;
    iteration = 0;
    check = true;
    
    % Elbow matrix
    VectorCodoDer = [CodoDerOrganizado(kd,1), CodoDerOrganizado(kd,2), CodoDerOrganizado(kd,3)];
    VEC_CODO_DER = [VEC_CODO_DER; VectorCodoDer];

    % End efector matrix
    MatrixDer = [MatrixDerRead(i,1) MatrixDerRead(i,2) MatrixDerRead(i,3) MatrixDerRead(i,4);
        MatrixDerRead(i+1,1) MatrixDerRead(i+1,2) MatrixDerRead(i+1,3) MatrixDerRead(i+1,4);
        MatrixDerRead(i+2,1) MatrixDerRead(i+2,2) MatrixDerRead(i+2,3) MatrixDerRead(i+2,4);
        MatrixDerRead(i+3,1) MatrixDerRead(i+3,2) MatrixDerRead(i+3,3) MatrixDerRead(i+3,4)];

    MatrixDer2 = inv(TD) * MatrixDer;

    MatrixDer(1:3,1:3)  = rotz(180) * MatrixDer(1:3,1:3);
    MatrixDer(1:2,4) = -MatrixDer(1:2,4);
    MatrixDer = TD * MatrixDer; 
    matrix = MatrixDer; % Para IK_UR

    Goal_Der = [Goal_Der; [MatrixDerRead(i,4) MatrixDerRead(i+1,4) MatrixDerRead(i+2,4)]];

    X_Goal2_Der = [MatrixDer2(1,4) MatrixDer2(2,4) MatrixDer2(3,4)]; 
    Goal2_Der = [Goal2_Der;X_Goal2_Der]; 
    Rot_Goal2_Der = [MatrixDer2(1,1) MatrixDer2(1,2) MatrixDer2(1,3); 
        MatrixDer2(2,1) MatrixDer2(2,2) MatrixDer2(2,3);
        MatrixDer2(3,1) MatrixDer2(3,2) MatrixDer2(3,3)];
    EulerAngles_Goal2_Der = rotm2eul(Rot_Goal2_Der); 
    Quat_Goal2_Der = eul2quat(EulerAngles_Goal2_Der);
    
    disp(' ')
    disp("------------- NEW RIGHT ARM ITERATION -------------")
    disp(kd)
   
    try
        IK_UR % Con MatrixDer
        %disp(res)
        
        for ii = 1:1:8
            for jj = 1:1:6 
                configSolnDer(jj).JointPosition = res(ii,jj);
            end       
            
            GoalConfigDer = [configSolnDer(1).JointPosition configSolnDer(2).JointPosition configSolnDer(3).JointPosition configSolnDer(4).JointPosition configSolnDer(5).JointPosition configSolnDer(6).JointPosition];
            [validState,~] = checkCollision(robotModelDer,GoalConfigDer',env,"IgnoreSelfCollision","off","Exhaustive","on","SkippedSelfCollisions","parent");
            if ~any(validState)
                %disp('No collisions ')
                % End efector for the specific configuration
                EfectorFinalDer = getTransform(robotDer,configSolnDer,'tool0','base_link');
                X_Der = [EfectorFinalDer(1,4) EfectorFinalDer(2,4) EfectorFinalDer(3,4)]; 
                Rot_Der =[EfectorFinalDer(1,1) EfectorFinalDer(1,2) EfectorFinalDer(1,3); 
                    EfectorFinalDer(2,1) EfectorFinalDer(2,2) EfectorFinalDer(2,3);
                    EfectorFinalDer(3,1) EfectorFinalDer(3,2) EfectorFinalDer(3,3)];
                EulerAngles_Der = rotm2eul(Rot_Der); 
                Quat_Der = eul2quat(EulerAngles_Der);

                % Wrist 1
                Wrist1Der = getTransform(robotDer,configSolnDer,'wrist_1_link','base_link');
                X_Wrist1Der = [Wrist1Der(1,4) Wrist1Der(2,4) Wrist1Der(3,4)]; % Guada las coordenadas de la muñeca

                % Wrist 2
                Wrist2Der = getTransform(robotDer,configSolnDer,'wrist_2_link','base_link');
                X_Wrist2Der = [Wrist2Der(1,4) Wrist2Der(2,4) Wrist2Der(3,4)];

                % Wrist 3
                Wrist3Der = getTransform(robotDer,configSolnDer,'wrist_3_link','base_link');
                X_Wrist3Der = [Wrist3Der(1,4) Wrist3Der(2,4) Wrist3Der(3,4)];

                % Elbow
                CodoDer = getTransform(robotDer,configSolnDer,'forearm_link','base_link');
                X_CodoDer = [CodoDer(1,4) CodoDer(2,4) CodoDer(3,4)];

                % Shoulder
                HombroSalienteDer = getTransform(robotDer,configSolnDer,'shoulder_link','base_link');
                X_HombroDer = [HombroSalienteDer(1,4) HombroSalienteDer(2,4) HombroSalienteDer(3,4)];
           
                % Error estimation
                DistXDer = distPosition(X_Der,X_Goal2_Der);
                DistODer = distOrientation(Quat_Der,Quat_Goal2_Der);
                %disp(DistODer)
                DistMDer = distanceMetric(VectorCodoDer, X_Wrist2Der, X_Wrist1Der, X_Wrist3Der, X_CodoDer, X_HombroDer, Goal_Der');
                %disp(DistMDer)

                % Wrist error estimation
                if i == 1 || ~exist('WristOldDer','var')
                    ErroWristDer = 0; 
                else
                    ErroWristDer = variationWrist(configSolnDer(4).JointPosition,WristOldDer); % Calcula el error como la diferencia entre el nuevo valor de la muñeca y el antiguo
                end

                DistDer = real(WX*DistXDer + WO*DistODer + WH*DistMDer + ErroWristDer);
                %disp(DistDer)

                CodoRobotRotDer = rotx(-45)*X_CodoDer';
                WristRobotRotDer= rotx(-45)*X_Wrist1Der';                

                % Check if it is finished
                if DistXDer <= 0.05 && DistODer <= 0.16 && ~(CodoRobotRotDer(2) > 0.1 && WristRobotRotDer(2) < CodoRobotRotDer(2)) && DistDer < DistMinDer %&& configSolnDer(1).JointPosition >= HombroLimDer(1) && configSolnDer(1).JointPosition <= HombroLimDer(2) %&& configSolnDer(2).JointPosition >= HombroLim2Der(1) && configSolnDer(2).JointPosition <= HombroLim2Der(2)
                    DistMinDer = DistDer;
                    MejorConfigDer = [configSolnDer(1).JointPosition configSolnDer(2).JointPosition configSolnDer(3).JointPosition configSolnDer(4).JointPosition configSolnDer(5).JointPosition configSolnDer(6).JointPosition];
                    mejor_ii = ii;
                end

            else
                %disp('Collision')
            end

        end

        if DistMinDer ~= DistR
            disp(' ')
            disp('Valid configuration found')
            WristOldDer = MejorConfigDer(1,4);
            MEJORES_DER = [MEJORES_DER;mejor_ii];
            ConfigFinalDer = [ConfigFinalDer;MejorConfigDer];
            ROTMAT_DER = [ROTMAT_DER;Rot_Goal2_Der];
        
        elseif exist('MejorConfigDer', 'var')
            disp(' ')
            disp('Not valid configuration found')
            ConfigFinalDer = [ConfigFinalDer;MejorConfigDer];
            WorstDer = kd;
            PEORES_DER = [PEORES_DER;WorstDer];

        else
            disp(' ')
            disp('Not any valid configuration found')
            WorstDer = kd;
            PEORES_DER = [PEORES_DER;WorstDer];            
        end
   
    catch
        disp(' ')
        disp('Wrong IK value')
    end     
end



% Filter the incorrect data and correct the position values
Goal_Izq(PEORES_IZQ,:) = [];
Goal2_Izq(PEORES_IZQ,:) = [];
Goal_Izq =[Goal_Izq(:,1)+0.07,Goal_Izq(:,2)+0.13,Goal_Izq(:,3)+1.15];

Goal_Der(PEORES_DER,:) = [];
Goal2_Der(PEORES_DER,:) = [];
Goal_Der =[Goal_Der(:,1)+0.07,Goal_Der(:,2)-0.13,Goal_Der(:,3)+1.15];


%% Plot the movements of the robotics arm respect the human elbow and wrist

COD_IZQ = []; % Vectores de posicion del codo del robot, sacado de las configuraciones
COD_DER = [];
END_IZQ = [];
END_DER = [];
ROTEND_IZQ = [];
ROTEND_DER = [];

robotIzq = loadrobot("universalUR3");
TIzq = [1 0 0 0.07;0 0.7071 0.7071 0.13;0 -0.7071 0.7071 1.15;0 0 0 1];
setFixedTransform(robotIzq.Bodies{1,1}.Joint,TIzq);
configuracionesIzq = robotIzq.homeConfiguration;

robotDer = loadrobot("universalUR3");
TDer = [1 0 0 0.07;0 0.7071 -0.7071 -0.13;0 0.7071 0.7071 1.15;0 0 0 1];
setFixedTransform(robotDer.Bodies{1,1}.Joint,TDer);
configuracionesDer = robotDer.homeConfiguration;

m=0;

for i=1:1:length(ConfigFinalIzq)
    m = m+1

    configuracionesIzq(1).JointPosition = ConfigFinalIzq(i,1); % Establece la ConfigFinal como la configuracion del robot
    configuracionesIzq(2).JointPosition = ConfigFinalIzq(i,2);
    configuracionesIzq(3).JointPosition = ConfigFinalIzq(i,3);
    configuracionesIzq(4).JointPosition = ConfigFinalIzq(i,4);
    configuracionesIzq(5).JointPosition = ConfigFinalIzq(i,5);
    configuracionesIzq(6).JointPosition = ConfigFinalIzq(i,6);

    configuracionesDer(1).JointPosition = ConfigFinalDer(i,1);
    configuracionesDer(2).JointPosition = ConfigFinalDer(i,2);
    configuracionesDer(3).JointPosition = ConfigFinalDer(i,3);
    configuracionesDer(4).JointPosition = ConfigFinalDer(i,4);
    configuracionesDer(5).JointPosition = ConfigFinalDer(i,5);
    configuracionesDer(6).JointPosition = ConfigFinalDer(i,6);

    figure(1);
    zoom(gca,'on'); % Zoom en los ejes actuales (get current axes)
    show(robotIzq,configuracionesIzq,"Collisions","off"); % Muestra el robot con la ConfigFinal
    view(90,30);
    camzoom(3);
    camtarget([0,0,0.5])
    hold on
    show(robotDer,configuracionesDer,"Collisions","off");
    hold on
    show(env{1});
    show(env{2});
    show(env{3});
    show(env{4});
    hold on

    Codo_Izq = getTransform(robotIzq,configuracionesIzq,'forearm_link','base_link'); 
    COD_IZQ = [COD_IZQ; Codo_Izq(1,4) Codo_Izq(2,4) Codo_Izq(3,4)]; % Guarda las posiciones del codo respecto a la base, el hombro
    Codo_Der = getTransform(robotDer,configuracionesDer,'forearm_link','base_link');
    COD_DER = [COD_DER; Codo_Der(1,4) Codo_Der(2,4) Codo_Der(3,4)];
    

    plot3(VEC_CODO_IZQ(m,1)+0.07, VEC_CODO_IZQ(m,2)+0.13, VEC_CODO_IZQ(m,3)+1.15,'o','Color','g','MarkerSize',10,'MarkerFaceColor','g')
    plot3(Goal_Izq(m,1),Goal_Izq(m,2),Goal_Izq(m,3),'o','Color','r','MarkerSize',10,'MarkerFaceColor','r')
    hold on

    plot3(VEC_CODO_DER(m,1)+0.07,VEC_CODO_DER(m,2)-0.13,VEC_CODO_DER(m,3)+1.15,'o','Color','g','MarkerSize',10,'MarkerFaceColor','g')
    plot3(Goal_Der(m,1),Goal_Der(m,2),Goal_Der(m,3),'o','Color','r','MarkerSize',10,'MarkerFaceColor','r')
    hold off

    PuntoEndIzq = getTransform(robotIzq,configuracionesIzq,'tool0','base_link');
    PuntoIzq = [Goal_Izq(m,1),Goal_Izq(m,2),Goal_Izq(m,3)];
    END_IZQ = [END_IZQ; PuntoEndIzq(1,4), PuntoEndIzq(2,4), PuntoEndIzq(3,4)];
    ROTEND_IZQ = [ROTEND_IZQ; PuntoEndIzq(1,1),PuntoEndIzq(1,2),PuntoEndIzq(1,3); PuntoEndIzq(2,1),PuntoEndIzq(2,2),PuntoEndIzq(2,3); PuntoEndIzq(3,1),PuntoEndIzq(3,2),PuntoEndIzq(3,3)];
    
    PuntoEndDer = getTransform(robotDer,configuracionesDer,'tool0','base_link');
    PuntoDer = [Goal_Der(m,1),Goal_Der(m,2),Goal_Der(m,3)];
    END_DER = [END_DER;PuntoEndDer(1,4),PuntoEndDer(2,4),PuntoEndDer(3,4)];
    ROTEND_DER = [ROTEND_DER;PuntoEndDer(1,1),PuntoEndDer(1,2),PuntoEndDer(1,3);PuntoEndDer(2,1),PuntoEndDer(2,2),PuntoEndDer(2,3);PuntoEndDer(3,1),PuntoEndDer(3,2),PuntoEndDer(3,3)];

    pause(0.00001);
    
end

%% Metrics

frechetIzq = frechet(Goal2_Izq, END_IZQ)
frechetDer = frechet(Goal2_Der, END_DER)

angularSimilarityIzq = angular_similarity(Goal2_Izq, END_IZQ)
angularSimilarityDer = angular_similarity(Goal2_Der, END_DER)

jerkIzqHuman = jerk(Goal2_Izq)
jerkIzqRobot = jerk(END_IZQ)
jerkDerHuman = jerk(Goal2_Der)
jerkDerRobot = jerk(END_DER)

%% Plot the movement in 3D (smoothed to visualized it better)

% Left
figure(3)
plot3(Goal2_Izq(:,1),Goal2_Izq(:,2),Goal2_Izq(:,3),'Color','g');
hold on
xlabel('X (m)')
ylabel('Y (m)')
zlabel('Z (m)')
plot3(END_IZQ(:,1),END_IZQ(:,2),END_IZQ(:,3),'Color','b');
plot3(Goal2_Izq(1,1),Goal2_Izq(1,2),Goal2_Izq(1,3),'o','Color','r','MarkerSize',5,'MarkerFaceColor','r');
plot3(Goal2_Izq(length(Goal2_Izq),1),Goal2_Izq(length(Goal2_Izq),2),Goal2_Izq(length(Goal2_Izq),3),'o','Color','m','MarkerSize',5,'MarkerFaceColor','m');
legend('Robot data','Human data','Initial point','Final point')
hold off
title('Left', 'FontSize', 14)

%Right
figure(4)
plot3(Goal2_Der(:,1),Goal2_Der(:,2),Goal2_Der(:,3),'Color','g');
hold on
xlabel('X (m)')
ylabel('Y (m)')
zlabel('Z (m)')
plot3(END_DER(:,1),END_DER(:,2),END_DER(:,3),'Color','b');
plot3(Goal2_Der(1,1),Goal2_Der(1,2),Goal2_Der(1,3),'o','Color','r','MarkerSize',5,'MarkerFaceColor','r');
plot3(Goal2_Der(length(Goal2_Der),1),Goal2_Der(length(Goal2_Der),2),Goal2_Der(length(Goal2_Der),3),'o','Color','m','MarkerSize',5,'MarkerFaceColor','m');
legend('Robot data','Human data','Initial point','Final point')
hold off
title('Right', 'FontSize', 14)


%% Plot the error values of the end efector 

THETA_IZQ = [];
THETA2_IZQ = [];
V_IZQ = [];
V2_IZQ = [];
THETA_DER = [];
THETA2_DER = [];
V_DER = [];
V2_DER = [];

for i=1:3:length(ROTEND_IZQ)

    R1_IZQ = [ROTMAT_IZQ(i,1),ROTMAT_IZQ(i,2),ROTMAT_IZQ(i,3);ROTMAT_IZQ(i+1,1),ROTMAT_IZQ(i+1,2),ROTMAT_IZQ(i+1,3);ROTMAT_IZQ(i+2,1),ROTMAT_IZQ(i+2,2),ROTMAT_IZQ(i+2,3)];
    R2_IZQ = [ROTEND_IZQ(i,1),ROTEND_IZQ(i,2),ROTEND_IZQ(i,3);ROTEND_IZQ(i+1,1),ROTEND_IZQ(i+1,2),ROTEND_IZQ(i+1,3);ROTEND_IZQ(i+2,1),ROTEND_IZQ(i+2,2),ROTEND_IZQ(i+2,3)];
    R_1_IZQ = R1_IZQ;
    R_2_IZQ = R2_IZQ;
    ang_axis_izq = rotm2axang(R_1_IZQ); % Guarda la salida de la función en una variable
    ang_axis2_izq = rotm2axang(R_2_IZQ);
    theta_izq = ang_axis_izq(4); % Ángulo de rotación en radianes
    theta2_izq = ang_axis2_izq(4);
    v_izq = ang_axis_izq(1:3); % Coordenadas del eje de rotación unitario
    v2_izq = ang_axis2_izq(1:3);
    THETA_IZQ = [THETA_IZQ;theta_izq];
    THETA2_IZQ = [THETA2_IZQ;theta2_izq];
    V_IZQ = [V_IZQ;v_izq];
    V2_IZQ = [V2_IZQ;v2_izq];

end

for i=1:3:length(ROTEND_DER)
    R1_DER = [ROTMAT_DER(i,1),ROTMAT_DER(i,2),ROTMAT_DER(i,3);ROTMAT_DER(i+1,1),ROTMAT_DER(i+1,2),ROTMAT_DER(i+1,3);ROTMAT_DER(i+2,1),ROTMAT_DER(i+2,2),ROTMAT_DER(i+2,3)];
    R2_DER = [ROTEND_DER(i,1),ROTEND_DER(i,2),ROTEND_DER(i,3);ROTEND_DER(i+1,1),ROTEND_DER(i+1,2),ROTEND_DER(i+1,3);ROTEND_DER(i+2,1),ROTEND_DER(i+2,2),ROTEND_DER(i+2,3)];
    R_1_DER = R1_DER;
    R_2_DER = R2_DER;
    ang_axis_der = rotm2axang(R_1_DER); % Guarda la salida de la función en una variable
    ang_axis2_der = rotm2axang(R_2_DER);
    theta_der = ang_axis_der(4); % Ángulo de rotación en radianes
    theta2_der = ang_axis2_der(4);
    v_der = ang_axis_der(1:3); % Coordenadas del eje de rotación unitario
    v2_der = ang_axis2_der(1:3);
    THETA_DER = [THETA_DER;theta_der];
    THETA2_DER = [THETA2_DER;theta2_der];
    V_DER = [V_DER;v_der];
    V2_DER = [V2_DER;v2_der];    
end


% Left
figure(5)

subplot(3,2,1);
plot(Goal2_Izq(:,1),'Color','g');
hold on
plot(END_IZQ(:,1),'Color','r');
legend("X Robot","X Human");
hold off
title('Left X');
xlabel('Index');
ylabel('Position X (m)');

subplot(3,2,2);
plot(Goal2_Izq(:,2),'Color','g');
hold on
plot(END_IZQ(:,2),'Color','r');
legend("Y Robot","Y Human");
hold off
title('Left Y');
xlabel('Index');
ylabel('Position Y (m)');

subplot(3,2,3);
plot(Goal2_Izq(:,3),'Color','g');
hold on
plot(END_IZQ(:,3),'Color','r');
legend("Z Robot","Z Human");
hold off
title('Left Z');
xlabel('Index');
ylabel('Position Z (m)');

subplot(3,2,4);
boxplot(abs(Goal2_Izq(:,:) - END_IZQ(:,:)),'Labels',{'Error X','Error Y','Error Z'});
title('Position Error');
ylabel('Error (m)');

subplot(3,2,5);
plot(THETA_IZQ);
hold on
plot(THETA2_IZQ)
title('Rotation end-effector');
xlabel('Index');
ylabel('Theta (rad)');
legend("Theta Robot","Theta Human");
hold off

subplot(3,2,6);
boxplot(abs(THETA_IZQ - THETA2_IZQ)*0.3);
title('Error Theta');
ylabel('Theta error (rad)');
hold off

% Right
figure(6)

subplot(3,2,1);
plot(Goal2_Der(:,1),'Color','g');
hold on
plot(END_DER(:,1),'Color','r');
legend("X Robot","X Human");
hold off
title('Right X');
xlabel('Index');
ylabel('Position X (m)');

subplot(3,2,2);
plot(Goal2_Der(:,2),'Color','g');
hold on
plot(END_DER(:,2),'Color','r');
legend("Y Robot","Y Human");
hold off
title('Right Y');
xlabel('Index');
ylabel('Position Y (m)');

subplot(3,2,3);
plot(Goal2_Der(:,3),'Color','g');
hold on
plot(END_DER(:,3),'Color','r');
legend("Z Robot","Z Human");
hold off
title('Right Z');
xlabel('Index');
ylabel('Position Z (m)');

subplot(3,2,4);
boxplot(abs(Goal2_Der(:,:) - END_DER(:,:)),'Labels',{'Error X','Error Y','Error Z'});
title('Position Error');
ylabel('Error (m)');

subplot(3,2,5);
plot(THETA_DER);
hold on
plot(THETA2_DER)
title('Rotation end-effector');
xlabel('Index');
ylabel('Theta (rad)');
legend("Theta Robot","Theta Human");
hold off

subplot(3,2,6);
boxplot(abs(THETA_DER - THETA2_DER)*0.3);
title('Error Theta');
ylabel('Theta error (rad)');
hold off

