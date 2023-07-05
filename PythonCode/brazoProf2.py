''' Code created by Adrian Prados and Blanca Lopez, 
researchers from RoboticsLab, University Carlos III of Madrid, Spain'''


#!/home/nox/anaconda3/envs/ikfastenv/bin/ python3
import sys
import pyrealsense2 as rs
import mediapipe as mp
import cv2
import numpy as np
import datetime as dt
import time
import math as mt
from scipy.spatial.transform import Rotation
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


################### Functions ##################################

def get_label(index, hand, results):
    for idx, classification in enumerate(results.multi_handedness):
        if classification.classification[0].index == index:
            label = classification.classification[0].label
            text = '{}'.format(label)
            coords = tuple(np.multiply(
                np.array((hand.landmark[mpHands.HandLandmark.WRIST].x, hand.landmark[mpHands.HandLandmark.WRIST].y)), [640, 480]).astype(int))

            return text, coords

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

def HandPlaneOrientation(points):
    ''' Obtain the zvector of the final efector as the ortogonal vector of the hand plane'''
    normal_vector = np.cross(points[0] - points[2], points[0] - points[1]) # Producto de vectores para obtener la normal
    normal_vector /= np.linalg.norm(normal_vector) # Lo divide por su norma para volverlo unitario
    x_vec = (points[2]-points[1])
    x_vec /= np.linalg.norm(x_vec)
    y_vec = np.cross(normal_vector,x_vec)
    y_vec /= np.linalg.norm(y_vec)

    ''' The -1 correct the orientation of the hand plane respect the image orientation'''
    Mat = np.matrix([
        [-1*x_vec[0],x_vec[1],x_vec[2]],
        [-1*y_vec[0],y_vec[1],y_vec[2]],
        [normal_vector[0],-1*normal_vector[1],-1*normal_vector[2]]
         ])

    angle = 90
    Rox = np.matrix([   # Rotación 90º en x
        [1, 0, 0],
        [0, mt.cos(mt.radians(angle)), -mt.sin(mt.radians(angle))],
        [0, mt.sin(mt.radians(angle)), mt.cos(mt.radians(angle))]
        ])   

    Roz = np.matrix([   # Rotación 90º en z
        [mt.cos(-mt.radians(angle)), -mt.sin(-mt.radians(angle)), 0],
        [mt.sin(-mt.radians(angle)), mt.cos(-mt.radians(angle)), 0],
        [0, 0, 1]
        ])


    Rotacional = np.matmul(Rox,Roz)
    Rotacional = np.linalg.inv(Rotacional)
    Mat = np.transpose(Mat)
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

def smooth_rotations(rotations, sigma=1):
    r11 = []
    r12 = []
    r13 = []
    r21 = []
    r22 = []
    r23 = []
    r31 = []
    r32 = []
    r33 = []
    for r in rotations:
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

def smooth_endefector(rotations, sigma=1):
    X = []
    Y= []
    Z = []
    for r in rotations:
        X.append(r[0,3])
        Y.append(r[1,3])
        Z.append(r[2,3])
    XEnd = gaussian_filter(X, sigma)
    YEnd = gaussian_filter(Y, sigma)
    ZEnd = gaussian_filter(Z, sigma)

    return XEnd,YEnd,ZEnd

def smooth_elbow(elbow, sigma=1):
    X = []
    Y= []
    Z = []
    for r in elbow:
        X.append(r[0,0])
        Y.append(r[1,0])
        Z.append(r[2,0])
    XElbow = gaussian_filter(X, sigma)
    YElbow = gaussian_filter(Y, sigma)
    ZElbow = gaussian_filter(Z, sigma)

    return XElbow,YElbow,ZElbow

def plot_smoothed_rotations(rotations, R1,R2,R3,R4,R5,R6,R7,R8,R9):
    rotations = np.array(rotations)  # Convertimos a arreglo NumPy
    fig, axs = plt.subplots(3, 3, figsize=(10,10))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    axs = axs.ravel()

    for i in range(9):
        if i < 3:
            axs[i].set_title("Row {}".format(i))
        if i % 3 == 0:
            axs[i].set_ylabel("Axis {}".format(i//3))

        axs[i].plot(rotations[:, i//3, i%3], label='raw')
        axs[i].legend()
    axs[0].plot(R1, label='filter')
    axs[0].legend()
    axs[1].plot(R2, label='filter')
    axs[1].legend()
    axs[2].plot(R3, label='filter')
    axs[2].legend()
    axs[3].plot(R4, label='filter')
    axs[3].legend()
    axs[4].plot(R5, label='filter')
    axs[4].legend()
    axs[5].plot(R6, label='filter')
    axs[5].legend()
    axs[6].plot(R7, label='filter')
    axs[6].legend()
    axs[7].plot(R8, label='filter')
    axs[7].legend()
    axs[8].plot(R9, label='filter')
    axs[8].legend()
    plt.show()

def plot_smoothed_EndEfector(rotations, X,Y,Z):
    rotations = np.array(rotations)  # Convertimos a arreglo NumPy
    fig, axs = plt.subplots(1, 3, figsize=(10,10))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    axs = axs.ravel()
    Xr = []
    Yr = []
    Zr = []

    for r in rotations:
        Xr.append(r[0,3])
        Yr.append(r[1,3])
        Zr.append(r[2,3])

    for i in range(3):
        if i < 3:
            axs[i].set_title("Row {}".format(i))

    axs[0].plot(Xr,label='raw')
    axs[0].plot(X, label='filter')
    axs[0].legend()
    axs[1].plot(Yr,label='raw')
    axs[1].plot(Y, label='filter')
    axs[1].legend()
    axs[2].plot(Zr,label='raw')
    axs[2].plot(Z, label='filter')
    axs[2].legend()
    plt.show()

def plot_smoothed_Elbow(elbow, X,Y,Z):
    elbow = np.array(elbow)  # Convertimos a arreglo NumPy
    fig, axs = plt.subplots(1, 3, figsize=(10,10))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    axs = axs.ravel()
    Xr = []
    Yr= []
    Zr = []
    for r in elbow:
        Xr.append(r[0,0])
        Yr.append(r[1,0])
        Zr.append(r[2,0])

    for i in range(3):
        if i < 3:
            axs[i].set_title("Row {}".format(i))

    axs[0].plot(Xr,label='raw')
    axs[0].plot(X, label='filter')
    axs[0].legend()
    axs[1].plot(Yr,label='raw')
    axs[1].plot(Y, label='filter')
    axs[1].legend()
    axs[2].plot(Zr,label='raw')
    axs[2].plot(Z, label='filter')
    axs[2].legend()
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
''' Characteristics for hands and pose tracking'''
#------- Hands ---------
''' Link for Hands: https://google.github.io/mediapipe/solutions/hands.html'''
mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.3) # The confidence can be change for the specific project
desired_solution = [0, 0, 0, 0, 0, 0] 
#------- Body --------
''' Link for BodyPose : https://google.github.io/mediapipe/solutions/pose.html'''
mpPose = mp.solutions.pose
pose = mpPose.Pose(min_detection_confidence=0.1)

mpDraw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# ====== Enable Streams ======
''' Activate the stream caracteristics for the RealSense D435i'''
config.enable_device(device)

#For worse FPS, but better resolution:

#stream_res_x = 1280
#stream_res_y = 720

#For better FPS. but worse resolution:

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
clipping_distance_in_meters = 2
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
    results = hands.process(color_images_rgb)
    cuerpo = pose.process(color_images_rgb)

    ''' Load the intrinsics values of the camera RealSense D435i'''
    INTR = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
    
    ''' If body is detected check the hands'''
    if cuerpo.pose_landmarks and results.multi_hand_landmarks:

        ''' Draw body lines and save body references'''
        mpDraw.draw_landmarks(images, cuerpo.pose_landmarks, mpPose.POSE_CONNECTIONS)

        codo_der = cuerpo.pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_ELBOW]
        muneca_der = cuerpo.pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_WRIST]
        hombro_der = cuerpo.pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_SHOULDER]

        codo_izq = cuerpo.pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_ELBOW]
        muneca_izq = cuerpo.pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_WRIST]
        hombro_izq = cuerpo.pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_SHOULDER]

        ''' calculate the angle of the elbow (just to check)'''
        angulo_der = calculate_angle([hombro_der.x,hombro_der.y], [codo_der.x,codo_der.y], [muneca_der.x,muneca_der.y])
        angulo_izq = calculate_angle([hombro_izq.x,hombro_izq.y], [codo_izq.x,codo_izq.y], [muneca_izq.x,muneca_izq.y])

        #--------- Human Elbows --------
        codo_der_X = int(codo_der.x*len(depth_image_flipped[0])) #Lo multiplica por el tamaño de la imagen pq la coordenada x esta normalizada entre 0 y 1
        codo_der_Y = int(codo_der.y*len(depth_image_flipped))
        if codo_der_X >= len(depth_image_flipped[0]):
            codo_der_X = len(depth_image_flipped[0]) - 1
        if codo_der_Y>= len(depth_image_flipped):
            codo_der_Y = len(depth_image_flipped) - 1

        codo_izq_X = int(codo_izq.x*len(depth_image_flipped[0]))
        codo_izq_Y = int(codo_izq.y*len(depth_image_flipped))
        if codo_izq_X >= len(depth_image_flipped[0]):
            codo_izq_X = len(depth_image_flipped[0]) - 1
        if codo_izq_Y>= len(depth_image_flipped):
            codo_izq_Y = len(depth_image_flipped) - 1

        # ----------- Human Wrists --------
        muneca_der_X = int(muneca_der.x*len(depth_image_flipped[0]))  
        muneca_der_Y = int(muneca_der.y*len(depth_image_flipped))
        if muneca_der_X >= len(depth_image_flipped[0]):
            muneca_der_X = len(depth_image_flipped[0]) - 1
        if muneca_der_Y >= len(depth_image_flipped):
            muneca_der_Y = len(depth_image_flipped) - 1

        muneca_izq_X = int(muneca_izq.x*len(depth_image_flipped[0]))
        muneca_izq_Y = int(muneca_izq.y*len(depth_image_flipped))
        if muneca_izq_X >= len(depth_image_flipped[0]):
            muneca_izq_X = len(depth_image_flipped[0]) - 1
        if muneca_izq_Y >= len(depth_image_flipped):
            muneca_izq_Y = len(depth_image_flipped) - 1

        # ----------- Human Shoulders ---------    
        hombro_der_X = int(hombro_der.x*len(depth_image_flipped[0]))
        hombro_der_Y = int(hombro_der.y*len(depth_image_flipped))
        if hombro_der_X  >= len(depth_image_flipped[0]):
            hombro_der_X  = len(depth_image_flipped[0]) - 1
        if hombro_der_Y >= len(depth_image_flipped):
            hombro_der_Y = len(depth_image_flipped) - 1

        hombro_izq_X = int(hombro_izq.x*len(depth_image_flipped[0]))
        hombro_izq_Y = int(hombro_izq.y*len(depth_image_flipped))
        if hombro_izq_X  >= len(depth_image_flipped[0]):
            hombro_izq_X  = len(depth_image_flipped[0]) - 1
        if hombro_izq_Y >= len(depth_image_flipped):
            hombro_izq_Y = len(depth_image_flipped) - 1


        ''' Z values for the elbow, wrist and shoulder'''
        Z_CODO_DER = depth_image_flipped[codo_der_Y,codo_der_X] * depth_scale # meters
        Z_MUNECA_DER = depth_image_flipped[muneca_der_Y,muneca_der_X] * depth_scale # meters
        Z_HOMBRO_DER = depth_image_flipped[hombro_der_Y,hombro_der_X] * depth_scale # meters
        Z_CODO_IZQ = depth_image_flipped[codo_izq_Y,codo_izq_X] * depth_scale # meters
        Z_MUNECA_IZQ = depth_image_flipped[muneca_izq_Y,muneca_izq_X] * depth_scale # meters
        Z_HOMBRO_IZQ = depth_image_flipped[hombro_izq_Y,hombro_izq_X] * depth_scale # meters

        '''Values of the different studied points in meters'''
        Codo_der_Bueno = rs.rs2_deproject_pixel_to_point(INTR,[codo_der_X,codo_der_Y],Z_CODO_DER)
        Muneca_der_Bueno = rs.rs2_deproject_pixel_to_point(INTR,[muneca_der_X,muneca_der_Y],Z_MUNECA_DER)
        Hombro_der_Bueno = rs.rs2_deproject_pixel_to_point(INTR,[hombro_der_X,hombro_der_Y],Z_HOMBRO_DER)
        Codo_izq_Bueno = rs.rs2_deproject_pixel_to_point(INTR,[codo_izq_X,codo_izq_Y],Z_CODO_IZQ)
        Muneca_izq_Bueno = rs.rs2_deproject_pixel_to_point(INTR,[muneca_izq_X,muneca_izq_Y],Z_MUNECA_IZQ)
        Hombro_izq_Bueno = rs.rs2_deproject_pixel_to_point(INTR,[hombro_izq_X,hombro_izq_Y],Z_HOMBRO_IZQ)
        
        ''' Calculate the rotation of the left shoulder to orientate correctly to the camera the body'''
        theta = mt.atan2((Hombro_der_Bueno[2]-Hombro_izq_Bueno[2]),(Hombro_der_Bueno[0]-Hombro_izq_Bueno[0]))
        theta = 180 - mt.degrees(theta)# Angle to rotate the left arm
        # As we rotate using the Y axis, if rigth shoulder is the nearest to te camera, the angle is negative
        if theta > 180:
            theta = -1*(360 - theta)
        else:
            theta = theta
        #images = cv2.putText(images, f"THETA: {theta}", org5, font, fontScale, color, thickness, cv2.LINE_AA)

        '''Generates the rotation for all the points'''
        Pivote = (Hombro_izq_Bueno[0],Hombro_izq_Bueno[2])
        RotHombroDer = (Hombro_der_Bueno[0],Hombro_der_Bueno[2])
        RotHombroIzq = (Hombro_izq_Bueno[0],Hombro_izq_Bueno[2])
        RotCodoDer = (Codo_der_Bueno[0],Codo_der_Bueno[2])
        RotCodoIzq = (Codo_izq_Bueno[0],Codo_izq_Bueno[2])
        RotMunecaDer= (Muneca_der_Bueno[0],Muneca_der_Bueno[2])
        RotMunecaIzq= (Muneca_izq_Bueno[0],Muneca_izq_Bueno[2])
        rotadoHD = rotateY(Pivote,RotHombroDer,mt.radians(theta))
        rotadoHI = rotateY(Pivote,RotHombroIzq,mt.radians(theta))
        rotadoCD = rotateY(Pivote,RotCodoDer,mt.radians(theta))
        rotadoCI = rotateY(Pivote,RotCodoIzq,mt.radians(theta))
        rotadoMD = rotateY(Pivote,RotMunecaDer,mt.radians(theta))
        rotadoMI = rotateY(Pivote,RotMunecaIzq,mt.radians(theta))

        HombroDerFinal= [rotadoHD[0],Hombro_der_Bueno[1],rotadoHD[1]]
        HombroIzqFinal= [rotadoHI[0],Hombro_izq_Bueno[1],rotadoHI[1]]
        CodoDerFinal= [rotadoCD[0],Codo_der_Bueno[1],rotadoCD[1]]
        CodoIzqFinal= [rotadoCI[0],Codo_izq_Bueno[1],rotadoCI[1]]
        MunecaDerFinal= [rotadoMD[0],Muneca_der_Bueno[1],rotadoMD[1]]
        MunecaIzqFinal= [rotadoMI[0],Muneca_izq_Bueno[1],rotadoMI[1]]

        ''' Reduction factor for the human arm'''
        HumanHumeroDer = abs(mt.dist(HombroDerFinal,CodoDerFinal))
        if HumanHumeroDer < 0.26 or HumanHumeroDer > 0.28:
            CodoDerFinal = project_point_on_line(HombroDerFinal,CodoDerFinal,0.27)
        HumanHumeroIzq = abs(mt.dist(HombroIzqFinal,CodoIzqFinal))
        if HumanHumeroIzq < 0.26 or HumanHumeroIzq > 0.28:
            CodoIzqFinal = project_point_on_line(HombroIzqFinal,CodoIzqFinal,0.27)

        HumanCubitoDer = abs(mt.dist(CodoDerFinal,MunecaDerFinal))
        if HumanCubitoDer < 0.23 or HumanCubitoDer > 0.25:
            MunecaDerFinal = project_end_efector(CodoDerFinal,MunecaDerFinal,0.24)
        HumanCubitoIzq = abs(mt.dist(CodoIzqFinal,MunecaIzqFinal))
        if HumanCubitoIzq < 0.23 or HumanCubitoIzq > 0.25:
            MunecaIzqFinal = project_end_efector(CodoIzqFinal,MunecaIzqFinal,0.24)
        
        BrazoHumanDer = HumanHumeroDer + HumanCubitoDer
        BrazoHumanIzq = HumanHumeroIzq + HumanCubitoIzq
        

        if BrazoHumanIzq <=0.8:
            try:
                ''' obtain factor Robot-human for arm length'''
                factorRH_izq = (0.5/BrazoHumanIzq)
            except:
                factorRH_izq = 1

            '''Calculate the translation between points of the human to the robot base reference
            Human to robot reference -> yhuman = zbase; xhuman=ybase; zhuman = xbase;'''

            robotHombroIzq = [0,0,0] 
            Translation = [(robotHombroIzq[0] + HombroIzqFinal[2]),(robotHombroIzq[1] + HombroIzqFinal[0]),(robotHombroIzq[2] + HombroIzqFinal[1])]

            robotHombroIzq = [(Translation[0] - HombroIzqFinal[2]),(Translation[1] - HombroIzqFinal[0]),(Translation[2]- HombroIzqFinal[1])]
            robotCodoIzq = [(Translation[0] - CodoIzqFinal[2])*factorRH_izq,(Translation[1] - CodoIzqFinal[0])*factorRH_izq,(Translation[2] - CodoIzqFinal[1])*factorRH_izq]
            robotMunecaIzq = [(Translation[0] - MunecaIzqFinal[2])*factorRH_izq,(Translation[1] - MunecaIzqFinal[0])*factorRH_izq,(Translation[2] - MunecaIzqFinal[1])*factorRH_izq]

            ''' Detection of left hand orientation'''
            if results.multi_handedness[0].classification[0].label == 'Left':    

                for num, hand in enumerate(results.multi_hand_landmarks): 
                    mpDraw.draw_landmarks(images, hand, mpHands.HAND_CONNECTIONS)

                    if get_label(num, hand, results): 
                        text, coord = get_label(num, hand, results) 
                        cv2.putText(images, text, coord, font, 1, (255, 0, 0), 2, cv2.LINE_AA)

                for handLms in results.multi_hand_landmarks:

                    Cero_izq = results.multi_hand_landmarks[0].landmark[mpHands.HandLandmark.WRIST]
                    Cinco_izq = results.multi_hand_landmarks[0].landmark[mpHands.HandLandmark.INDEX_FINGER_MCP]
                    Diecisiete_izq = results.multi_hand_landmarks[0].landmark[mpHands.HandLandmark.PINKY_MCP]
                    
                    '''Depth hand values for the 0,5,17 hand reference plane'''
                    Cero_izq_X = int(Cero_izq.x*len(depth_image_flipped[0]))
                    Cero_izq_Y = int(Cero_izq.y*len(depth_image_flipped))
                    if Cero_izq_X >= len(depth_image_flipped[0]):
                        Cero_izq_X = len(depth_image_flipped[0]) - 1

                    if Cero_izq_Y>= len(depth_image_flipped):
                        Cero_izq_Y = len(depth_image_flipped) - 1

                    Cinco_izq_X = int(Cinco_izq.x*len(depth_image_flipped[0]))
                    Cinco_izq_Y = int(Cinco_izq.y*len(depth_image_flipped))
                    if Cinco_izq_X >= len(depth_image_flipped[0]):
                        Cinco_izq_X = len(depth_image_flipped[0]) - 1

                    if Cinco_izq_Y>= len(depth_image_flipped):
                        Cinco_izq_Y = len(depth_image_flipped) - 1

                    Diecisiete_izq_X = int(Diecisiete_izq.x*len(depth_image_flipped[0]))
                    Diecisiete_izq_Y = int(Diecisiete_izq.y*len(depth_image_flipped))
                    if Diecisiete_izq_X >= len(depth_image_flipped[0]):
                        Diecisiete_izq_X = len(depth_image_flipped[0]) - 1

                    if Diecisiete_izq_Y>= len(depth_image_flipped):
                        Diecisiete_izq_Y = len(depth_image_flipped) - 1

                    ''' Z values for the left hand (depth)'''
                    Z_CERO_IZQ = depth_image_flipped[Cero_izq_Y,Cero_izq_X] * depth_scale # meters
                    Z_CINCO_IZQ = depth_image_flipped[Cinco_izq_Y,Cinco_izq_X] * depth_scale # meters
                    Z_DIECISITE_IZQ = depth_image_flipped[Diecisiete_izq_Y,Diecisiete_izq_X] * depth_scale # meters

                    '''Values of the different studied points in meters'''
                    CeroArrayIzq = rs.rs2_deproject_pixel_to_point(INTR,[Cero_izq_X,Cero_izq_Y],Z_CERO_IZQ)
                    CincoArrayIzq = rs.rs2_deproject_pixel_to_point(INTR,[Cinco_izq_X,Cinco_izq_Y],Z_CINCO_IZQ)
                    DiecisieteArrayIzq = rs.rs2_deproject_pixel_to_point(INTR,[Diecisiete_izq_X,Diecisiete_izq_Y],Z_DIECISITE_IZQ)
                
                points_izq = np.asarray([CeroArrayIzq, CincoArrayIzq, DiecisieteArrayIzq])
                #print(points_izq)
                MatRot_izq = HandPlaneOrientation(points_izq)
                #print(MatRot_izq)
            
            
            ''' Generate the values for the UR3 robot'''
            #m=np.array([[1,0,0,-0.07],[0,0.7072,-0.7072,0.7214],[0,0.7072,0.7072,-0.9052],[0,0,0,1]])
            punto_izq = [robotMunecaIzq[0],robotMunecaIzq[1],robotMunecaIzq[2],1]
            puntoCodo_izq = np.array([[robotCodoIzq[0]],[robotCodoIzq[1]],[robotCodoIzq[2]]])
            
            try:
                MatrizBrazoIzq = np.array([
                    [round(MatRot_izq[0,0],2),round(MatRot_izq[0,1],2),round(MatRot_izq[0,2],2),punto_izq[0]],
                    [round(MatRot_izq[1,0],2),round(MatRot_izq[1,1],2),round(MatRot_izq[1,2],2),punto_izq[1]],
                    [round(MatRot_izq[2,0],2),round(MatRot_izq[2,1],2),round(MatRot_izq[2,2],2),punto_izq[2]],
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
                            CORCODOPRE_IZQ.append(puntoCodo_izq)
                            print("Valor de codo izquierdo",puntoCodo_izq[0,0])
                            r1 = Rotation.from_matrix(MatRot_izq)
                            angles1 = r1.as_euler("xyz",degrees=False)
                            efectorFinal_izq = [MatrizBrazoIzq[0,3],MatrizBrazoIzq[1,3],MatrizBrazoIzq[2,3],angles1[0],angles1[1],angles1[2]]
                            EFECTOR_IZQ.append(efectorFinal_izq)
                        h = h + 1


                datos_izq = datos_izq + 1

            except ValueError:
                print("Mathematical inconsistence")
                
        else:
            print("Incorrect value")
        
        
        if BrazoHumanDer <=0.8:
            try:
                ''' obtain factor Robot-human for arm length'''
                factorRH_der = (0.5/BrazoHumanDer)
            except:
                factorRH_der = 1
        
            '''Calculate the translation between points of the human to the robot base reference
            Human to robot reference -> yhuman = zbase; xhuman=ybase; zhuman = xbase;'''

            robotHombroDer = [0,0,0] 
            Translation = [(robotHombroDer[0] + HombroDerFinal[2]),(robotHombroDer[1] + HombroDerFinal[0]),(robotHombroDer[2] + HombroDerFinal[1])]

            robotHombroDer = [(Translation[0] - HombroDerFinal[2]),(Translation[1] - HombroDerFinal[0]),(Translation[2]- HombroDerFinal[1])]
            robotCodoDer = [(Translation[0] - CodoDerFinal[2])*factorRH_der,(Translation[1] - CodoDerFinal[0])*factorRH_der,(Translation[2] - CodoDerFinal[1])*factorRH_der]
            robotMunecaDer = [(Translation[0] - MunecaDerFinal[2])*factorRH_der,(Translation[1] - MunecaDerFinal[0])*factorRH_der,(Translation[2] - MunecaDerFinal[1])*factorRH_der]
            
            
            ''' Detection of right hand orientation'''
            if results.multi_handedness[0].classification[0].label == 'Right':   

                for num, hand in enumerate(results.multi_hand_landmarks): 
                    mpDraw.draw_landmarks(images, hand, mpHands.HAND_CONNECTIONS)

                    if get_label(num, hand, results): 
                        text, coord = get_label(num, hand, results) 
                        cv2.putText(images, text, coord, font, 1, (255, 0, 0), 2, cv2.LINE_AA) 

                for handLms in results.multi_hand_landmarks:
                    mpDraw.draw_landmarks(images, handLms, mpHands.HAND_CONNECTIONS)
                    Cero_der = results.multi_hand_landmarks[0].landmark[mpHands.HandLandmark.WRIST]
                    Cinco_der = results.multi_hand_landmarks[0].landmark[mpHands.HandLandmark.INDEX_FINGER_MCP]
                    Diecisiete_der = results.multi_hand_landmarks[0].landmark[mpHands.HandLandmark.PINKY_MCP]
                    
                    '''Depth hand values for the 0,5,17 hand reference plane'''
                    Cero_der_X = int(Cero_der.x*len(depth_image_flipped[0]))
                    Cero_der_Y = int(Cero_der.y*len(depth_image_flipped))
                    if Cero_der_X >= len(depth_image_flipped[0]):
                        Cero_der_X = len(depth_image_flipped[0]) - 1

                    if Cero_der_Y>= len(depth_image_flipped):
                        Cero_der_Y = len(depth_image_flipped) - 1

                    Cinco_der_X = int(Cinco_der.x*len(depth_image_flipped[0]))
                    Cinco_der_Y = int(Cinco_der.y*len(depth_image_flipped))
                    if Cinco_der_X >= len(depth_image_flipped[0]):
                        Cinco_der_X = len(depth_image_flipped[0]) - 1

                    if Cinco_der_Y>= len(depth_image_flipped):
                        Cinco_der_Y = len(depth_image_flipped) - 1

                    Diecisiete_der_X = int(Diecisiete_der.x*len(depth_image_flipped[0]))
                    Diecisiete_der_Y = int(Diecisiete_der.y*len(depth_image_flipped))
                    if Diecisiete_der_X >= len(depth_image_flipped[0]):
                        Diecisiete_der_X = len(depth_image_flipped[0]) - 1

                    if Diecisiete_der_Y>= len(depth_image_flipped):
                        Diecisiete_der_Y = len(depth_image_flipped) - 1

                    ''' Z values for the left hand (depth)'''
                    Z_CERO_DER = depth_image_flipped[Cero_der_Y,Cero_der_X] * depth_scale # meters
                    Z_CINCO_DER = depth_image_flipped[Cinco_der_Y,Cinco_der_X] * depth_scale # meters
                    Z_DIECISITE_DER = depth_image_flipped[Diecisiete_der_Y,Diecisiete_der_X] * depth_scale # meters

                    '''Values of the different studied points in meters'''
                    CeroArrayDer = rs.rs2_deproject_pixel_to_point(INTR,[Cero_der_X,Cero_der_Y],Z_CERO_DER)
                    CincoArrayDer = rs.rs2_deproject_pixel_to_point(INTR,[Cinco_der_X,Cinco_der_Y],Z_CINCO_DER)
                    DiecisieteArrayDer = rs.rs2_deproject_pixel_to_point(INTR,[Diecisiete_der_X,Diecisiete_der_Y],Z_DIECISITE_DER)
                
                points_der = np.asarray([CeroArrayDer, CincoArrayDer, DiecisieteArrayDer])
                MatRot_der = HandPlaneOrientation(points_der)
                print(MatRot_der)

            ''' Generate the values for the UR3 robot'''
            #m=np.array([[1,0,0,-0.07],[0,0.7072,-0.7072,0.7214],[0,0.7072,0.7072,-0.9052],[0,0,0,1]])
            punto_der = [robotMunecaDer[0],robotMunecaDer[1],robotMunecaDer[2],1]
            puntoCodo_der= np.array([[robotCodoDer[0]],[robotCodoDer[1]],[robotCodoDer[2]]])
            
            try:
                MatrizBrazoDer = np.array([
                    [round(MatRot_der[0,0],2),round(MatRot_der[0,1],2),round(MatRot_der[0,2],2),punto_der[0]],
                    [round(MatRot_der[1,0],2),round(MatRot_der[1,1],2),round(MatRot_der[1,2],2),punto_der[1]],
                    [round(MatRot_der[2,0],2),round(MatRot_der[2,1],2),round(MatRot_der[2,2],2),punto_der[2]],
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
                            CORCODOPRE.append(puntoCodo_der)
                            print("Valor de codo derecho",puntoCodo_der[0,0])
                            r2 = Rotation.from_matrix(MatRot_der)
                            angles2 = r2.as_euler("xyz",degrees=False)
                            efectorFinal_der = [MatrizBrazoDer[0,3],MatrizBrazoDer[1,3],MatrizBrazoDer[2,3],angles2[0],angles2[1],angles2[2]]
                            EFECTOR_DER.append(efectorFinal_der)
                        j = j + 1


                datos_der = datos_der + 1

            except ValueError:
                print("Mathematical inconsistence")
                
        else:
            print("Incorrect value")

    
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
        print("User pressed break key for SN:",device)
        break

'''Filter orientation using a Gaussian Filter'''
print(DATOSPRE_IZQ)
print(DATOSPRE_DER)

L1,L2,L3,L4,L5,L6,L6,L8,L9 = smooth_rotations(DATOSPRE_IZQ,1)
R1,R2,R3,R4,R5,R6,R7,R8,R9 = smooth_rotations(DATOSPRE_DER,1) # Filter values [0.5, 1]

X_End_Izq,Y_End_Izq,Z_End_Izq = smooth_endefector(DATOSPRE_IZQ,1)
X_End_Der,Y_End_Der,Z_End_Der = smooth_endefector(DATOSPRE_DER,1)
X_Elbow_Izq,Y_Elbow_Izq,Z_Elbow_Izq = smooth_elbow(CORCODOPRE_IZQ,1)
X_Elbow_Der,Y_Elbow_Der,Z_Elbow_Der = smooth_elbow(CORCODOPRE_DER,1)
#plot_smoothed_Elbow(CORCODOPRE,XElbow,YElbow,ZElbow)

print("**********************")
#print(R1)
#plot_smoothed_rotations(DATOSPRE,R1,R2,R3,R4,R5,R6,R7,R8,R9)

'''Save data filtered'''
#print(DATOSPRE[0][0,0])
for n in range(len(L1)):
    
    MatrizFiltered = np.array([
        [L1[n],L2[n],L3[n],X_End_Izq[n]],
        [L4[n],L5[n],L6[n],Y_End_Izq[n]],
        [L7[n],L8[n],L9[n],Z_End_Izq[n]],
        [0,0,0,1]
        ], np.float64)
    DATOS_IZQ.append(MatrizFiltered)

for n in range(len(R1)):
    
    MatrizFiltered = np.array([
        [R1[n],R2[n],R3[n],X_End_Der[n]],
        [R4[n],R5[n],R6[n],Y_End_Der[n]],
        [R7[n],R8[n],R9[n],Z_End_Der[n]],
        [0,0,0,1]
        ], np.float64)
    DATOS_DER.append(MatrizFiltered)


for n in range(len(X_Elbow_Izq)):
    puntoCodoFilter = np.array([[X_Elbow_Izq[n]],[Y_Elbow_Izq[n]],[Z_Elbow_Izq[n]]])
    CORCODO_IZQ.append(puntoCodoFilter)

for n in range(len(X_Elbow_Der)):
    puntoCodoFilter = np.array([[X_Elbow_Der[n]],[Y_Elbow_Der[n]],[Z_Elbow_Der[n]]])
    CORCODO_DER.append(puntoCodoFilter)


print("--------------------------")
print(DATOS_IZQ)
print(DATOS_DER)
print("--------------------------")


''' Save all the values in .csv'''
variable = np.asarray(DATOS_IZQ).shape
print("DATOS IZQ: ",variable[0])
DATOS_IZQ = np.reshape(DATOS_IZQ, (variable[0]*4, -1))
print(np.asarray(DATOS_IZQ).shape)
Modelo_izq = pd.DataFrame(DATOS_IZQ)
Modelo_izq.to_csv('/home/carlos/TAICHI/HumanData/DatosBrazoIzquierdo.csv',index=False, header=False) 

variable2 = np.asarray(DATOS_DER).shape
print("DATOS DER: ",variable2[0])
DATOS_DER = np.reshape(DATOS_DER, (variable2[0]*4, -1))
print(np.asarray(DATOS_DER).shape)
Modelo_der = pd.DataFrame(DATOS_DER)
Modelo_der.to_csv('/home/carlos/TAICHI/HumanData/DatosBrazoDerecho.csv',index=False, header=False) 

variable3 = np.asarray(CORCODO_IZQ).shape
CORCODO_IZQ= np.reshape(CORCODO_IZQ, (variable3[0]*3, -1))
ModeloCodo_izq = pd.DataFrame(CORCODO_IZQ)
ModeloCodo_izq.to_csv('/home/carlos/TAICHI/HumanData/CodoIzquierdo.csv',index=False, header=False)

variable4 = np.asarray(CORCODO_DER).shape
CORCODO_DER= np.reshape(CORCODO_DER, (variable4[0]*3, -1))
ModeloCodo_der = pd.DataFrame(CORCODO_DER)
ModeloCodo_der.to_csv('/home/carlos/TAICHI/HumanData/CodoDerecho.csv',index=False, header=False)

ModeloEfectorFinalIzq = pd.DataFrame(EFECTOR_IZQ)
ModeloEfectorFinalIzq.to_csv('/home/carlos/TAICHI/HumanData/EfectorFinalIzquierdo.csv',index=False, header=False)

ModeloEfectorFinalDer = pd.DataFrame(EFECTOR_DER)
ModeloEfectorFinalDer.to_csv('/home/carlos/TAICHI/HumanData/EfectorFinalDerecho.csv',index=False, header=False)

''' Close the application'''
print("Application Closing")
pipeline.stop()
print("Application Closed.")
#plot_smoothed_rotations(DATOSPRE,R1,R2,R3,R4,R5,R6,R7,R8,R9)

