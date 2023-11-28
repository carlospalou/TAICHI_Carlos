import pyrealsense2 as rs
import mediapipe as mp
import cv2
import numpy as np
import datetime as dt
import math as mt
from scipy.spatial.transform import Rotation
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


################### Functions ##################################

def rotateY(origin, point, angle):
    ''' Function defined to rotate one point respect to naother point'''
    #Angle in radians
    ox,oz = origin
    px,pz = point

    qx = ox + mt.cos(angle) * (px - ox) - mt.sin(angle) * (pz - oz)
    qz = oz + mt.sin(angle) * (px - ox) + mt.cos(angle) * (pz - oz)
    return qx, qz

def calculate_angle(a,b,c):
    ''' Function defined to calculate angle between three points'''
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

def HandPlaneOrientation(points, hand):
    ''' Obtain the Z vector of the final efector as the ortogonal vector of the hand plane'''
    z_vec = np.cross(points[0] - points[2], points[0] - points[1])
    x_vec = (points[2]-points[1])
    z_vec /= np.linalg.norm(z_vec) # Lo divide por su norma para volverlo unitario
    x_vec /= np.linalg.norm(x_vec)
    y_vec = np.cross(z_vec,x_vec)
    y_vec /= np.linalg.norm(y_vec)

    #print(z_vec, hand) 

    angle = 90

    if hand == 0:
        # The -1 correct the orientation of the hand plane respect the image orientation
        Mat = np.matrix([
            [-1*x_vec[0],-1*y_vec[0],z_vec[0]], 
            [x_vec[1],y_vec[1],-1*z_vec[1]],
            [x_vec[2],y_vec[2],-1*z_vec[2]]
            ])

        Rox = np.matrix([   # Rotación 90º en x
            [1, 0, 0],
            [0, mt.cos(mt.radians(angle)), -mt.sin(mt.radians(angle))],
            [0, mt.sin(mt.radians(angle)), mt.cos(mt.radians(angle))]
            ])   
        Roz = np.matrix([   # Rotación -90º en z
            [mt.cos(-mt.radians(angle)), -mt.sin(-mt.radians(angle)), 0],
            [mt.sin(-mt.radians(angle)), mt.cos(-mt.radians(angle)), 0],
            [0, 0, 1]
            ])

    if hand == 1:
        Mat = np.matrix([
            [-1*x_vec[0],-1*y_vec[0],z_vec[0]], 
            [x_vec[1],y_vec[1],-1*z_vec[1]],
            [x_vec[2],y_vec[2],-1*z_vec[2]]
            ])
        Rox = np.matrix([   # Rotación 90º en x
            [1, 0, 0],
            [0, mt.cos(mt.radians(angle)), -mt.sin(mt.radians(angle))],
            [0, mt.sin(mt.radians(angle)), mt.cos(mt.radians(angle))]
            ])
        Roz = np.matrix([   # Rotación 90º en z
            [mt.cos(-mt.radians(angle)), mt.sin(-mt.radians(angle)), 0],
            [-mt.sin(-mt.radians(angle)), mt.cos(-mt.radians(angle)), 0],
            [0, 0, 1]
            ])
        
    Rotacional = np.matmul(Rox,Roz)
    Rotacional = np.linalg.inv(Rotacional)
    MatRot = np.matmul(Rotacional,Mat)

    return MatRot

def project_point_on_line(A, B, distance):
    ''' Function that project the elbow in line to rectified the oclusions'''
    # Calculate the direction vector between A and B
    direction = np.array(B) - np.array(A)
    # Normalize the direction vector
    direction = direction / np.linalg.norm(direction)
    # Multiple the vector for the distance
    projection = A + direction * distance
    return projection

def project_end_efector(A, B, distance):
    ''' Function that project the elbow in line to rectified the oclusions'''
    # Calculate the direction vector between A and B
    direction = np.array(B) - np.array(A)
    # Normalize the direction vector
    direction = direction / np.linalg.norm(direction)
    # Multiple the vector for the distance
    projectionEndEfector = A + direction * distance
    return projectionEndEfector

def smooth_rotations(data, sigma=1):
    r11 = []
    r12 = []
    r13 = []
    r21 = []
    r22 = []
    r23 = []
    r31 = []
    r32 = []
    r33 = []

    for r in data:
        r11.append(r[0,0])
        r12.append(r[0,1])
        r13.append(r[0,2])
        r21.append(r[1,0])
        r22.append(r[1,1])
        r23.append(r[1,2])
        r31.append(r[2,0])
        r32.append(r[2,1])
        r33.append(r[2,2])

    R1 = gaussian_filter(r11, sigma)
    R2 = gaussian_filter(r12, sigma)
    R3 = gaussian_filter(r13, sigma)
    R4 = gaussian_filter(r21, sigma)
    R5 = gaussian_filter(r22, sigma)
    R6 = gaussian_filter(r23, sigma)
    R7 = gaussian_filter(r31, sigma)
    R8 = gaussian_filter(r32, sigma)
    R9 = gaussian_filter(r33, sigma)

    return R1,R2,R3,R4,R5,R6,R7,R8,R9

def smooth_endeffector(data, sigma=1):
    X = []
    Y = []
    Z = []

    for r in data:
        X.append(r[0,3])
        Y.append(r[1,3])
        Z.append(r[2,3])

    XEnd = gaussian_filter(X, sigma)
    YEnd = gaussian_filter(Y, sigma)
    ZEnd = gaussian_filter(Z, sigma)

    return XEnd,YEnd,ZEnd

def smooth_elbow(elbow, sigma=1):
    X = []
    Y = []
    Z = []

    for r in elbow:
        X.append(r[0,0])
        Y.append(r[1,0])
        Z.append(r[2,0])

    XElbow = gaussian_filter(X, sigma)
    YElbow = gaussian_filter(Y, sigma)
    ZElbow = gaussian_filter(Z, sigma)

    return XElbow,YElbow,ZElbow

def plot_smoothed_rotations(datosIzq,datosDer,L1,L2,L3,L4,L5,L6,L7,L8,L9,R1,R2,R3,R4,R5,R6,R7,R8,R9):
    datosIzq = np.array(datosIzq)  
    datosDer = np.array(datosDer)
    fig, axs = plt.subplots(3, 6, figsize=(20,10))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    axs = axs.ravel() # Permite acceder a la matriz de subgráficos mediante un solo índice

    for i in range(18):
        if i == 1:
            axs[i].set_title("Left\n\nRow {}".format(i))
        elif i == 4:
            axs[i].set_title("Right\n\nRow {}".format(i-3))
        elif i < 3:
            axs[i].set_title("Row {}".format(i))
        elif i < 6:
            axs[i].set_title("Row {}".format(i-3))
        if i % 6 == 0:
            axs[i].set_ylabel("Axis {}".format(i//6))

    axs[0].plot(L1, label='filter')
    axs[0].plot(datosIzq[:,0,0], label='raw')
    axs[0].legend()
    axs[1].plot(L2, label='filter')
    axs[1].plot(datosIzq[:,0,1], label='raw')
    axs[1].legend()
    axs[2].plot(L3, label='filter')
    axs[2].plot(datosIzq[:,0,2], label='raw')
    axs[2].legend()

    axs[3].plot(R1, label='filter')
    axs[3].plot(datosDer[:,0,0], label='raw')
    axs[3].legend()
    axs[4].plot(R2, label='filter')
    axs[4].plot(datosDer[:,0,1], label='raw')
    axs[4].legend()
    axs[5].plot(R3, label='filter')
    axs[5].plot(datosIzq[:,0,2], label='raw')
    axs[5].legend()

    axs[6].plot(L4, label='filter')
    axs[6].plot(datosIzq[:,1,0], label='raw')
    axs[6].legend()
    axs[7].plot(L5, label='filter')
    axs[7].plot(datosIzq[:,1,1], label='raw')
    axs[7].legend()
    axs[8].plot(L6, label='filter')
    axs[8].plot(datosIzq[:,1,2], label='raw')
    axs[8].legend()

    axs[9].plot(R4, label='filter')
    axs[9].plot(datosDer[:,1,0], label='raw')
    axs[9].legend()
    axs[10].plot(R5, label='filter')
    axs[10].plot(datosDer[:,1,1], label='raw')
    axs[10].legend()
    axs[11].plot(R6, label='filter')
    axs[11].plot(datosDer[:,1,2], label='raw')
    axs[11].legend()

    axs[12].plot(L7, label='filter')
    axs[12].plot(datosIzq[:,2,0], label='raw')
    axs[12].legend()
    axs[13].plot(L8, label='filter')
    axs[13].plot(datosIzq[:,2,1], label='raw')
    axs[13].legend()
    axs[14].plot(L9, label='filter')
    axs[14].plot(datosIzq[:,2,2], label='raw')
    axs[14].legend()

    axs[15].plot(R7, label='filter')
    axs[15].plot(datosDer[:,2,0], label='raw')
    axs[15].legend()
    axs[16].plot(R8, label='filter')
    axs[16].plot(datosDer[:,2,1], label='raw')
    axs[16].legend()
    axs[17].plot(R9, label='filter')
    axs[17].plot(datosDer[:,2,2], label='raw')
    axs[17].legend()

    fig.suptitle("Rotations", fontsize=16)
    plt.show()

def plot_smoothed_EndEffector(datosIzq,datosDer,X_Izq,Y_Izq,Z_Izq,X_Der,Y_Der,Z_Der):
    datosIzq = np.array(datosIzq)
    datosDer = np.array(datosDer)   
    fig, axs = plt.subplots(2, 3, figsize=(10,10))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    axs = axs.ravel()

    X_Izq_r = []
    Y_Izq_r = []
    Z_Izq_r = []
    X_Der_r = []
    Y_Der_r = []
    Z_Der_r = []

    for r in datosIzq:
        X_Izq_r.append(r[0,3])
        Y_Izq_r.append(r[1,3])
        Z_Izq_r.append(r[2,3])

    for r in datosDer:
        X_Der_r.append(r[0,3])
        Y_Der_r.append(r[1,3])
        Z_Der_r.append(r[2,3])

    for i in range(6):
        if i == 1:
            axs[i].set_title("Left\nRow {}".format(i))
        elif i < 3:
            axs[i].set_title("Row {}".format(i))
        elif i == 4:
            axs[i].set_title("Right\nRow {}".format(i-3))
        else:
            axs[i].set_title("Row {}".format(i-3))

    axs[0].plot(X_Izq_r,label='raw')
    axs[0].plot(X_Izq, label='filter')
    axs[0].legend()
    axs[1].plot(Y_Izq_r,label='raw')
    axs[1].plot(Y_Izq, label='filter')
    axs[1].legend()
    axs[2].plot(Z_Izq_r,label='raw')
    axs[2].plot(Z_Izq, label='filter')
    axs[2].legend()

    axs[3].plot(X_Der_r,label='raw')
    axs[3].plot(X_Der, label='filter')
    axs[3].legend()
    axs[4].plot(Y_Der_r,label='raw')
    axs[4].plot(Y_Der, label='filter')
    axs[4].legend()
    axs[5].plot(Z_Der_r,label='raw')
    axs[5].plot(Z_Der, label='filter')
    axs[5].legend()

    fig.suptitle("End Effector", fontsize=16)
    plt.show()

def plot_smoothed_Elbow(elbowIzq,elbowDer,X_Izq,Y_Izq,Z_Izq,X_Der,Y_Der,Z_Der):
    elbowIzq = np.array(elbowIzq)
    elbowDer = np.array(elbowDer)   
    fig, axs = plt.subplots(2, 3, figsize=(10,10))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    axs = axs.ravel()

    X_Izq_r = []
    Y_Izq_r = []
    Z_Izq_r = []
    X_Der_r = []
    Y_Der_r = []
    Z_Der_r = []

    for r in elbowIzq:
        X_Izq_r.append(r[0,0])
        Y_Izq_r.append(r[1,0])
        Z_Izq_r.append(r[2,0])

    for r in elbowDer:
        X_Der_r.append(r[0,0])
        Y_Der_r.append(r[1,0])
        Z_Der_r.append(r[2,0])

    for i in range(6):
        if i == 1:
            axs[i].set_title("Left\nRow {}".format(i))
        elif i < 3:
            axs[i].set_title("Row {}".format(i))
        elif i == 4:
            axs[i].set_title("Right\nRow {}".format(i-3))
        else:
            axs[i].set_title("Row {}".format(i-3))

    axs[0].plot(X_Izq_r,label='raw')
    axs[0].plot(X_Izq, label='filter')
    axs[0].legend()
    axs[1].plot(Y_Izq_r,label='raw')
    axs[1].plot(Y_Izq, label='filter')
    axs[1].legend()
    axs[2].plot(Z_Izq_r,label='raw')
    axs[2].plot(Z_Izq, label='filter')
    axs[2].legend()
    
    axs[3].plot(X_Der_r,label='raw')
    axs[3].plot(X_Der, label='filter')
    axs[3].legend()
    axs[4].plot(Y_Der_r,label='raw')
    axs[4].plot(Y_Der, label='filter')
    axs[4].legend()
    axs[5].plot(Z_Der_r,label='raw')
    axs[5].plot(Z_Der, label='filter')
    axs[5].legend()

    fig.suptitle("Elbow", fontsize=16)
    plt.show()


################################################################

# ======= VISUAL FONTS =======

font = cv2.FONT_HERSHEY_SIMPLEX
org = (20, 100)
fontScale = .5
color = (0,0,0)
thickness = 2

# ====== DATA ======
global DATOS_IZQ, DATOS_DER, DATOSPRE_IZQ, DATOSPRE_DER, CORCODO_IZQ, CORCODO_DER, CORCODOPRE_IZQ, CORCODOPRE_DER, EFECTOR_IZQ, EFECTOR_DER

DATOS_IZQ = []
DATOS_DER = []
DATOSPRE_IZQ = []
DATOSPRE_DER = []
CORCODO_IZQ = [] 
CORCODO_DER = []
CORCODOPRE_IZQ = []
CORCODOPRE_DER = []
EFECTOR_IZQ = []
EFECTOR_DER = []
datos_izq = 0
datos_der = 0
h = 0
j = 0

# ====== Realsense ======
''' Detect the camera RealSense D435i and activate it'''
realsense_ctx = rs.context()
connected_devices = [] # List of serial numbers for present cameras
for i in range(len(realsense_ctx.devices)):
    detected_camera = realsense_ctx.devices[i].get_info(rs.camera_info.serial_number)
    print("Detected_camera")
    connected_devices.append(detected_camera)
device = connected_devices[0] # For this code only one camera is neccesary
pipeline = rs.pipeline()
config = rs.config()
background_removed_color = 153 # Grey color for the background

# ====== Mediapipe ======
''' Link for BodyPose : https://google.github.io/mediapipe/solutions/pose.html'''
mpPose = mp.solutions.pose
pose = mpPose.Pose(min_detection_confidence=0.1)
mpDraw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ====== Enable Streams ======
''' Activate the stream caracteristics for the RealSense D435i'''
config.enable_device(device)

stream_res_x = 640
stream_res_y = 480

stream_fps = 30

config.enable_stream(rs.stream.depth, stream_res_x, stream_res_y, rs.format.z16, stream_fps)
config.enable_stream(rs.stream.color, stream_res_x, stream_res_y, rs.format.bgr8, stream_fps)
profile = pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)

# ====== Get depth Scale ======
''' Obtain the scale of the depth estimated by the depth sensors of the camara''' 
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale for Camera SN",device,"is: ",depth_scale)

# ====== Set clipping distance ======
''' Generate the maximun distance for the camera range to detect'''
clipping_distance_in_meters = 5
clipping_distance = clipping_distance_in_meters / depth_scale
print("Configuration Successful for SN", device)

# ====== Get and process images ====== 
print("Starting to capture images on SN:",device)

# ======= Algorithm =========
while True:
    start_time = dt.datetime.today().timestamp()

    '''Get and align frames from the camera to the point cloud'''
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)  #Aligns the depth frame and the color frames cause they are separated in the camera
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    
    if not aligned_depth_frame or not color_frame:
        print("\tNot aligned")
        continue


    ''' Process images to work with one color image'''
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    depth_image_flipped = cv2.flip(depth_image,1)
    color_image = np.asanyarray(color_frame.get_data())

    depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) # Depth image is 1 channel, while color image is 3
    background_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), background_removed_color, color_image)  # Where the condition is satisfied gets the values os backround_removed_color, elsewhere the ones from color_image 

    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    images = cv2.flip(background_removed,1)
    color_image = cv2.flip(color_image,1)
    color_images_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    ''' Process hands and pose estimations'''
    results = pose.process(color_images_rgb)

    ''' Load the intrinsics values of the camera RealSense D435i'''
    INTR = aligned_depth_frame.profile.as_video_stream_profile().intrinsics

    if results.pose_world_landmarks:

        mpDraw.draw_landmarks(images, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

        Hombro_izq_3D = results.pose_world_landmarks.landmark[mpPose.PoseLandmark.RIGHT_SHOULDER] # Directamente obtenemos los puntos con x, y, z, visibility
        Codo_izq_3D = results.pose_world_landmarks.landmark[mpPose.PoseLandmark.RIGHT_ELBOW]
        Muneca_izq_3D = results.pose_world_landmarks.landmark[mpPose.PoseLandmark.RIGHT_WRIST]
        
        Hombro_der_3D = results.pose_world_landmarks.landmark[mpPose.PoseLandmark.LEFT_SHOULDER]
        Codo_der_3D  = results.pose_world_landmarks.landmark[mpPose.PoseLandmark.LEFT_ELBOW]
        Muneca_der_3D = results.pose_world_landmarks.landmark[mpPose.PoseLandmark.LEFT_WRIST]

        #print(Muneca_der_3D)

        ''' Calculate the rotation of the left shoulder to orientate correctly to the camera'''
        theta = mt.atan2((Hombro_izq_3D.z-Hombro_der_3D.z),(Hombro_izq_3D.x-Hombro_der_3D.x))
        theta = 180 - mt.degrees(theta)  # As we rotate using the Y axis, if rigth shoulder is the nearest to te camera, the angle is negative
        if theta > 180:
            theta = -1*(360 - theta)
        else:
            theta = theta
        #images = cv2.putText(images, f"theta: {theta}", org, font, fontScale, (255, 0, 0), thickness, cv2.LINE_AA)


        '''Generates the rotation for all the points'''
        Pivote = (Hombro_der_3D.x,Hombro_der_3D.z) 
        
        rotar_hombro_izq = (Hombro_izq_3D.x,Hombro_izq_3D.z) 
        rotar_codo_izq = (Codo_izq_3D.x,Codo_izq_3D.z)
        rotar_muneca_izq= (Muneca_izq_3D.x,Muneca_izq_3D.z)

        rotar_hombro_der = (Hombro_der_3D.x,Hombro_der_3D.z)
        rotar_codo_der = (Codo_der_3D.x,Codo_der_3D.z)
        rotar_muneca_der= (Muneca_der_3D.x,Muneca_der_3D.z)
        
        hombro_izq_rotado = rotateY(Pivote,rotar_hombro_izq,mt.radians(theta)) 
        codo_izq_rotado = rotateY(Pivote,rotar_codo_izq,mt.radians(theta))
        muneca_izq_rotado = rotateY(Pivote,rotar_muneca_izq,mt.radians(theta))

        hombro_der_rotado = rotateY(Pivote,rotar_hombro_der,mt.radians(theta))
        codo_der_rotado = rotateY(Pivote,rotar_codo_der,mt.radians(theta))
        muneca_der_rotado = rotateY(Pivote,rotar_muneca_der,mt.radians(theta))

        Hombro_izq_Final= [hombro_izq_rotado[0],Hombro_izq_3D.y,hombro_izq_rotado[1]]  
        Codo_izq_Final= [codo_izq_rotado[0],Codo_izq_3D.y,codo_izq_rotado[1]]
        Muneca_izq_Final= [muneca_izq_rotado[0],Muneca_izq_3D.y,muneca_izq_rotado[1]]

        Hombro_der_Final= [hombro_der_rotado[0],Hombro_der_3D.y,hombro_der_rotado[1]]
        Codo_der_Final= [codo_der_rotado[0],Codo_der_3D.y,codo_der_rotado[1]]
        Muneca_der_Final= [muneca_der_rotado[0],Muneca_der_3D.y,muneca_der_rotado[1]]


        ''' Reduction factor for the human arm'''
        Human_Humero_izq = abs(mt.dist(Hombro_izq_Final,Codo_izq_Final))
        if Human_Humero_izq < 0.26 or Human_Humero_izq > 0.28:
            Codo_izq_Final = project_point_on_line(Hombro_izq_Final,Codo_izq_Final,0.27)

        Human_Cubito_izq = abs(mt.dist(Codo_izq_Final,Muneca_izq_Final))
        if Human_Cubito_izq < 0.23 or Human_Cubito_izq > 0.25:
            Muneca_izq_Final = project_end_efector(Codo_izq_Final,Muneca_izq_Final,0.24)

        Brazo_Human_izq = Human_Humero_izq + Human_Cubito_izq

        Human_Humero_der = abs(mt.dist(Hombro_der_Final,Codo_der_Final))
        if Human_Humero_der < 0.26 or Human_Humero_der > 0.28:
            Codo_der_Final = project_point_on_line(Hombro_der_Final,Codo_der_Final,0.27)

        Human_Cubito_der = abs(mt.dist(Codo_der_Final,Muneca_der_Final))
        if Human_Cubito_der < 0.23 or Human_Cubito_der > 0.25:
            Muneca_der_Final = project_end_efector(Codo_der_Final,Muneca_der_Final,0.24)
        
        Brazo_Human_der = Human_Humero_der + Human_Cubito_der
        
    
        if Brazo_Human_izq and Brazo_Human_der <= 0.8:

            try:
                #Obtain Robot-Human factor for arm length
                RH_factor_izq = (0.5/Brazo_Human_izq)
                RH_factor_der = (0.5/Brazo_Human_der)

            except:
                RH_factor_izq = 1
                RH_factor_der = 1

            '''Cambiamos del sistema de coordenadas de la cámara al sistema de coordenadas del robot: 
            x_base_izq = z_human; y_base_izq = x_human; z_base_izq = y_human'''

            Robot_Hombro_izq = [0,0,0] 
            Translation = [(Robot_Hombro_izq[0] + Hombro_izq_Final[2]),(Robot_Hombro_izq[1] + Hombro_izq_Final[0]),(Robot_Hombro_izq[2] + Hombro_izq_Final[1])]  

            Robot_Hombro_izq = [(Translation[0] - Hombro_izq_Final[2]),(Translation[1] - Hombro_izq_Final[0]),(Translation[2]- Hombro_izq_Final[1])]
            Robot_Codo_izq = [(Translation[0] - Codo_izq_Final[2])*RH_factor_izq,(Translation[1] - Codo_izq_Final[0])*RH_factor_izq,(Translation[2] - Codo_izq_Final[1])*RH_factor_izq]
            Robot_Muneca_izq = [(Translation[0] - Muneca_izq_Final[2])*RH_factor_izq,(Translation[1] - Muneca_izq_Final[0])*RH_factor_izq,(Translation[2] - Muneca_izq_Final[1])*RH_factor_izq]

            
            Robot_Hombro_der = [0,0,0] 
            Translation = [(Robot_Hombro_der[0] + Hombro_der_Final[2]),(Robot_Hombro_der[1] + Hombro_der_Final[0]),(Robot_Hombro_der[2] + Hombro_der_Final[1])]

            Robot_Hombro_der = [(Translation[0] - Hombro_der_Final[2]),(Translation[1] - Hombro_der_Final[0]),(Translation[2]- Hombro_der_Final[1])]
            Robot_Codo_der = [(Translation[0] - Codo_der_Final[2])*RH_factor_der,(Translation[1] - Codo_der_Final[0])*RH_factor_der,(Translation[2] - Codo_der_Final[1])*RH_factor_der]
            Robot_Muneca_der = [(Translation[0] - Muneca_der_Final[2])*RH_factor_der,(Translation[1] - Muneca_der_Final[0])*RH_factor_der,(Translation[2] - Muneca_der_Final[1])*RH_factor_der]


            '''Obtenemos los puntos de las manos '''

            Indice_izq_3D = results.pose_world_landmarks.landmark[mpPose.PoseLandmark.RIGHT_INDEX]
            Menique_izq_3D = results.pose_world_landmarks.landmark[mpPose.PoseLandmark.RIGHT_PINKY]
            Muneca_izq = [Muneca_izq_3D.x, Muneca_izq_3D.y, Muneca_izq_3D.z] # Cambiamos la manera en que se guardan las variables
            Indice_izq = [Indice_izq_3D.x, Indice_izq_3D.y, Indice_izq_3D.z]
            Menique_izq = [Menique_izq_3D.x, Menique_izq_3D.y, Menique_izq_3D.z]

            cv2.putText(images, "Left", (int(Muneca_izq_3D.x), int(Muneca_izq_3D.y)), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
            Indice_der_3D = results.pose_world_landmarks.landmark[mpPose.PoseLandmark.LEFT_INDEX]
            Menique_der_3D  = results.pose_world_landmarks.landmark[mpPose.PoseLandmark.LEFT_PINKY]
            Muneca_der = [Muneca_der_3D.x, Muneca_der_3D.y, Muneca_der_3D.z]
            Indice_der = [Indice_der_3D.x, Indice_der_3D.y, Indice_der_3D.z]
            Menique_der = [Menique_der_3D.x, Menique_der_3D.y, Menique_der_3D.z]

            cv2.putText(images, "Right", (int(Muneca_der_3D.x), int(Muneca_der_3D.y)), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
            
            #print(Menique_der)

            pointsIzq = np.asarray([Muneca_izq, Indice_izq, Menique_izq])
            MatRot_izq = HandPlaneOrientation(pointsIzq , 0) # 0 mano izquierda

            pointsDer = np.asarray([Muneca_der, Indice_der, Menique_der])
            MatRot_der = HandPlaneOrientation(pointsDer , 1) # 1 mano derecha

            
            ''' Generate the left values for the UR3 robot'''
            Punto_izq = [Robot_Muneca_izq[0],Robot_Muneca_izq[1],Robot_Muneca_izq[2],1] 
            PuntoCodo_izq = np.array([[Robot_Codo_izq[0]],[Robot_Codo_izq[1]],[Robot_Codo_izq[2]]])
            
            try:
                MatrizBrazoIzq = np.array([
                    [round(MatRot_izq[0,0],2),round(MatRot_izq[0,1],2),round(MatRot_izq[0,2],2),Punto_izq[0]],
                    [round(MatRot_izq[1,0],2),round(MatRot_izq[1,1],2),round(MatRot_izq[1,2],2),Punto_izq[1]],
                    [round(MatRot_izq[2,0],2),round(MatRot_izq[2,1],2),round(MatRot_izq[2,2],2),Punto_izq[2]],
                    [0,0,0,1]
                    ], np.float64)
            
                if datos_izq > 10:
                    if MatRot_izq[0,0]== "nan" or MatRot_izq[0,1]== "nan" or MatRot_izq[0,2]== "nan":
                        continue
                    else:
                        ''' Correct data is saved'''
                        if h == 1:
                            h = 0
                            DATOSPRE_IZQ.append(MatrizBrazoIzq)
                            CORCODOPRE_IZQ.append(PuntoCodo_izq)
                            print("Valor de codo izquierdo",PuntoCodo_izq[0,0])
                            r1 = Rotation.from_matrix(MatRot_izq)
                            angles1 = r1.as_euler("xyz",degrees=False)
                            EfectorFinal_izq = [MatrizBrazoIzq[0,3],MatrizBrazoIzq[1,3],MatrizBrazoIzq[2,3],angles1[0],angles1[1],angles1[2]]
                            EFECTOR_IZQ.append(EfectorFinal_izq)
                        h = h + 1

                datos_izq = datos_izq + 1

            except ValueError:
                    print("Mathematical inconsistence")

                
            ''' Generate the right values for the UR3 robot'''
            Punto_der = [Robot_Muneca_der[0],Robot_Muneca_der[1],Robot_Muneca_der[2],1]
            PuntoCodo_der= np.array([[Robot_Codo_der[0]],[Robot_Codo_der[1]],[Robot_Codo_der[2]]])
            
            try:
                MatrizBrazoDer = np.array([
                    [round(MatRot_der[0,0],2),round(MatRot_der[0,1],2),round(MatRot_der[0,2],2),Punto_der[0]],
                    [round(MatRot_der[1,0],2),round(MatRot_der[1,1],2),round(MatRot_der[1,2],2),Punto_der[1]],
                    [round(MatRot_der[2,0],2),round(MatRot_der[2,1],2),round(MatRot_der[2,2],2),Punto_der[2]],
                    [0,0,0,1]
                    ], np.float64)
            
                if datos_der > 10:
                    if MatRot_der[0,0]== "nan" or MatRot_der[0,1]== "nan" or MatRot_der[0,2]== "nan":
                        continue
                    else:
                        ''' Correct data is saved'''
                        if j == 1:
                            j = 0
                            DATOSPRE_DER.append(MatrizBrazoDer)
                            CORCODOPRE_DER.append(PuntoCodo_der)
                            print("Valor de codo derecho",PuntoCodo_der[0,0])
                            r2 = Rotation.from_matrix(MatRot_der)
                            angles2 = r2.as_euler("xyz",degrees=False)
                            EfectorFinal_der = [MatrizBrazoDer[0,3],MatrizBrazoDer[1,3],MatrizBrazoDer[2,3],angles2[0],angles2[1],angles2[2]]
                            EFECTOR_DER.append(EfectorFinal_der)
                        j = j + 1

                datos_der = datos_der + 1

            except ValueError:
                print("Mathematical inconsistence")

        else:
            print("Incorrect arm value")

    else:
        print("No person in front of the camera")


    ''' Visualization of the image'''
    name_of_window = 'SN: ' + str(device)

    '''Display images''' 
    cv2.namedWindow(name_of_window, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(name_of_window, images)

    key = cv2.waitKey(1)
    # Press esc or 'q' to close the image window
    if key & 0xFF == ord('q') or key == 27:
        #print("User pressed break key for SN:",device)
        break


'''Filter orientation using a Gaussian Filter'''
#print(DATOSPRE_IZQ)
#print(DATOSPRE_DER)

L1,L2,L3,L4,L5,L6,L7,L8,L9 = smooth_rotations(DATOSPRE_IZQ,1)
R1,R2,R3,R4,R5,R6,R7,R8,R9 = smooth_rotations(DATOSPRE_DER,1) 

X_End_Izq,Y_End_Izq,Z_End_Izq = smooth_endeffector(DATOSPRE_IZQ,1)
X_End_Der,Y_End_Der,Z_End_Der = smooth_endeffector(DATOSPRE_DER,1)

X_Elbow_Izq,Y_Elbow_Izq,Z_Elbow_Izq = smooth_elbow(CORCODOPRE_IZQ,1)
X_Elbow_Der,Y_Elbow_Der,Z_Elbow_Der = smooth_elbow(CORCODOPRE_DER,1)

print("**********************")

'''Save data filtered'''
for n in range(len(L1)):
    
    MatrizIzqFiltered = np.array([
        [L1[n],L2[n],L3[n],X_End_Izq[n]],
        [L4[n],L5[n],L6[n],Y_End_Izq[n]],
        [L7[n],L8[n],L9[n],Z_End_Izq[n]],
        [0,0,0,1]
        ], np.float64)
    DATOS_IZQ.append(MatrizIzqFiltered)

for n in range(len(R1)):
    
    MatrizDerFiltered = np.array([
        [R1[n],R2[n],R3[n],X_End_Der[n]],
        [R4[n],R5[n],R6[n],Y_End_Der[n]],
        [R7[n],R8[n],R9[n],Z_End_Der[n]],
        [0,0,0,1]
        ], np.float64)
    DATOS_DER.append(MatrizDerFiltered)


for n in range(len(X_Elbow_Izq)):
    PuntoCodoFilterIzq = np.array([[X_Elbow_Izq[n]],[Y_Elbow_Izq[n]],[Z_Elbow_Izq[n]]])
    CORCODO_IZQ.append(PuntoCodoFilterIzq)

for n in range(len(X_Elbow_Der)):
    PuntoCodoFilterDer = np.array([[X_Elbow_Der[n]],[Y_Elbow_Der[n]],[Z_Elbow_Der[n]]])
    CORCODO_DER.append(PuntoCodoFilterDer)


#print("--------------------------")
#print(DATOS_IZQ)
#print(DATOS_DER)
#print("--------------------------")


''' Save all the values in .csv'''
# DATOS_IZQ es una lista donde cada elemento es una matriz de transformación homogénea 4x4
variable = np.asarray(DATOS_IZQ).shape # Convierte DATOS_IZQ en un array de numpy, .shape da las dimensiones del array
#print("DATOS IZQ: ",variable) # Número de matrices de transformacion homogeneas
DATOS_IZQ = np.reshape(DATOS_IZQ, (variable[0]*4, -1)) # Reorganiza el array anterior en uno de 4 columnas con las matrices una debajo de otra
#print(np.asarray(DATOS_IZQ).shape)
ModeloIzq = pd.DataFrame(DATOS_IZQ) # Crea una tabla o data frame de Pandas
ModeloIzq.to_csv('/home/carlos/TAICHI_Carlos/HumanData/Prueba5/DatosBrazoIzquierdo.csv',index=False, header=False) # Guarda la tabla con las matrices en un .csv

variable2 = np.asarray(DATOS_DER).shape
DATOS_DER = np.reshape(DATOS_DER, (variable2[0]*4, -1))
ModeloDer = pd.DataFrame(DATOS_DER)
ModeloDer.to_csv('/home/carlos/TAICHI_Carlos/HumanData/Prueba5/DatosBrazoDerecho.csv',index=False, header=False) 

# CORCODO_IZQ es una lista donde cada elemento es una matriz 3x1 con las coordenadas del codo izquierdo
variable3 = np.asarray(CORCODO_IZQ).shape 
CORCODO_IZQ= np.reshape(CORCODO_IZQ, (variable3[0]*3, -1)) # Reorganiza el array anterior en uno de una única columna y tres veces el número de filas que de puntos
#print(CORCODO_IZQ)
ModeloCodoIzq = pd.DataFrame(CORCODO_IZQ)
ModeloCodoIzq.to_csv('/home/carlos/TAICHI_Carlos/HumanData/Prueba5/CodoIzquierdo.csv',index=False, header=False)

variable4 = np.asarray(CORCODO_DER).shape
CORCODO_DER= np.reshape(CORCODO_DER, (variable4[0]*3, -1))
ModeloCodoDer = pd.DataFrame(CORCODO_DER)
ModeloCodoDer.to_csv('/home/carlos/TAICHI_Carlos/HumanData/Prueba5/CodoDerecho.csv',index=False, header=False)

ModeloEfectorFinalIzq = pd.DataFrame(EFECTOR_IZQ)
ModeloEfectorFinalIzq.to_csv('/home/carlos/TAICHI_Carlos/HumanData/Prueba5/EfectorFinalIzquierdo.csv',index=False, header=False)

ModeloEfectorFinalDer = pd.DataFrame(EFECTOR_DER)
ModeloEfectorFinalDer.to_csv('/home/carlos/TAICHI_Carlos/HumanData/Prueba5/EfectorFinalDerecho.csv',index=False, header=False)

''' Close the application'''
pipeline.stop()
print("Application Closed.")


#plot_smoothed_rotations(DATOSPRE_IZQ,DATOSPRE_DER,L1,L2,L3,L4,L5,L6,L7,L8,L9,R1,R2,R3,R4,R5,R6,R7,R8,R9)
plot_smoothed_EndEffector(DATOSPRE_IZQ,DATOSPRE_DER,X_End_Izq,Y_End_Izq,Z_End_Izq,X_End_Der,Y_End_Der,Z_End_Der)
#plot_smoothed_Elbow(CORCODOPRE_IZQ,CORCODOPRE_DER,X_Elbow_Izq,Y_Elbow_Izq,Z_Elbow_Izq,X_Elbow_Der,Y_Elbow_Der,Z_Elbow_Der)