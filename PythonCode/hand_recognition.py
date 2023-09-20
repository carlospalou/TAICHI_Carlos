import mediapipe as mp 
import cv2
import numpy as np
import uuid
import os
import math as mt
import sys
import openpose as op

params = dict()
params["logging_level"] = 3
params["output_resolution"] = "-1x-1"
params["net_resolution"] = "-1x368"
params["model_pose"] = "BODY_25"
params["alpha_pose"] = 0.6
params["scale_gap"] = 0.3
params["scale_number"] = 1
params["render_threshold"] = 0.05
params["num_gpu_start"] = 0
params["disable_blending"] = False
params["default_model_folder"] = "/home/kenghee/openpose/models/"
openpose = op.OpenPose(params)

frame = cv2.imread("/home/kenghee/openpose/examples/media/h5.jpg")

# one person, only one left hand
hands_rectangles = [[[200, 150, 428, 390], [0, 0, 0, 0]]]

for box in hands_rectangles[0]:
    cv2.rectangle(frame, (box[0],box[1]), (box[2],box[3]), (77, 255, 9), 3, 1)

left_hands, right_hands,frame = openpose.forward_hands(frame, hands_rectangles, True)

print ("left hands:")
print (left_hands)

while 1:
    cv2.imshow("output", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


'''
def get_label(index, hand, results):
    for idx, classification in enumerate(results.multi_handedness):
        if classification.classification[0].index == index:

            label = classification.classification[0].label
            text = '{}'.format(label)
            #score = classification.classification[0].score
            #text = '{} {}'.format(label, round(score, 2))

            coords = tuple(np.multiply(
                np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x, hand.landmark[mp_hands.HandLandmark.WRIST].y)), [640, 480]).astype(int))

            return text, coords


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

hands = mp_hands.Hands()
    
while cap.isOpened():
    ret, frame = cap.read()

    #BGR 2 RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image = cv2.flip(image, 1)

    image.flags.writeable = False

    #Detections
    results = hands.process(image)

    image.flags.writeable = True

    #RGB 2 BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    #Drawing
    if results.multi_hand_landmarks: 
        for num, hand in enumerate(results.multi_hand_landmarks): #Iterates through the landmarks (hand) and saves the indexs in num
            mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)

            if get_label(num, hand, results): #Check if we get something back
                text, coord = get_label(num, hand, results) #Save the outputs of the function in variables
                cv2.putText(image, text, coord, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Hand tracking', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):

        print(hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP])

        break

cap.release()
cv2.destroyAllWindows()
'''

