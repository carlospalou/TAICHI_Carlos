import mediapipe as mp 
import cv2
import numpy as np
import uuid
import os
import math as mt
import sys

#git test

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

    if results.multi_handedness[0].classification[0].label == 'Left':
        print('Left')
    
    if results.multi_handedness[0].classification[0].label == 'Left':
        print('Right')

    cv2.imshow('Hand tracking', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):

        print(hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP])

        break

cap.release()
cv2.destroyAllWindows()
