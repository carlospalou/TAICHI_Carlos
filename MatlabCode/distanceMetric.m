%% Function to evaluate the distance between the human elbow and the rest of the robotic joints
function [D] = distanceMetric(CodoH,W2,W1,W3,X_Codo,Hombro,Goal)

    %% Option 1 
%     C_W2 = norm(CodoH-W2);
%     C_W1 = norm(CodoH-W1);
%     C_W3 = norm(CodoH-W3);
%     C_CR = norm(CodoH-X_Codo);
%     C_HR = norm(CodoH-Hombro);
%     D = 10*C_W1+300*C_CR+C_HR; % That cost function must be adapted for your problem 

    %% Option 2
    X_Codo_Rot = rotx(45)*X_Codo';
    PuntoMedioBrazo = (Goal./norm(Goal)).*(norm(Goal)/2);

    DD = norm(X_Codo_Rot-PuntoMedioBrazo);
    
    D = 300*DD;
end

