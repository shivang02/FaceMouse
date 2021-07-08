"""
Creates a GUI window displaying user's face getting tracked in Real Time
"""
# Importing packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
from pyautogui import size
import time
import dlib
import cv2
import mouse
import threading
import math

# Initializing indexes for the features to track as an Ordered Dictionary
FACIAL_LANDMARKS_IDXS = OrderedDict([
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
])


def shape_arr_func(shape, dtype="int"):
    """
    Function to convert shape of facial landmark to a 2-tuple numpy array
    """
    # Initializing list of coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # Looping over the 68 facial landmarks and converting them
    # to a 2-tuple of (x, y) coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # Returning the list of (x, y) coordinates
    return coords


def mvmt_func(x):
    """
    Function to calculate the move value as fractional power of displacement.
    This helps to reduce noise in the motion
    """
    if x > 1.:
        return math.pow(x, 3.0 / 2.0)
    elif x < -1.:
        return -math.pow(abs(x), 3.0 / 2.0)
    elif 0. < x < 1.:
        return 1.0
    elif -1. < x < 0.:
        return -1.0
    else:
        return 0.0


def ear_func(eye):
    """
    Function to calculate the Eye Aspect Ratio.
    """
    # Finding the euclidean distance between two groups of vertical eye landmarks [(x, y) coords]
    v1 = dist.euclidean(eye[1], eye[5])
    v2 = dist.euclidean(eye[2], eye[4])
    # Finding the euclidean distance between the horizontal eye landmarks [(x, y) coords]
    h = dist.euclidean(eye[0], eye[3])
    # Finding the Eye Aspect Ratio (E.A.R)
    ear = (v1 + v2) / (2.0 * h)
    # Returning the Eye Aspect Ratio (E.A.R)
    return ear


# Defining a constant to indicate a blink when the EAR gets less than the threshold
# Next two constants to specify the number of frames blink has to be sustained
EYE_AR_THRESH = 0.20
EYE_AR_CONSEC_FRAMES_MIN = 1
EYE_AR_CONSEC_FRAMES_MAX = 5

# Initializing Frame COUNTER and the TOTAL number of blinks in a go
COUNTER = 0
TOTAL = 0

# Initializing Mouse Down Toggle and Click Type
isMouseDown = False
click_type = ""

# Initializing Dlib's face detector (HOG-based) and creating the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Taking the indexes of left eye, right eye and nose
(lStart, lEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]
(nStart, nEnd) = FACIAL_LANDMARKS_IDXS["nose"]

# Initializing the Video Capture from source
vs = cv2.VideoCapture(1)
# 1 sec pause to load the Video Stream before running the predictor
time.sleep(1.0)


def left_click_func():
    """
    Function to handle left clicks via blinking
    """

    global isMouseDown
    global click_type
    global TOTAL
    # Performs a mouse up event if blink is observed after mouse down event
    if isMouseDown and TOTAL != 0:
        mouse.release(button='left')
        isMouseDown = False
        click_type = "Mouse Up"

    else:
        # Single Click
        if TOTAL == 1:
            mouse.click(button='left')
            click_type = "Single Click"
        # Double Click
        elif TOTAL == 2:
            mouse.double_click(button='left')
            click_type = "Double Click"
        # Mouse Down (to drag / scroll)
        elif TOTAL == 3:
            mouse.press(button='left')
            isMouseDown = True
            click_type = "Mouse Down"
    # Resetting the TOTAL number of blinks counted in a go
    TOTAL = 0


def right_click_func():
    """
    Function to perform right click triggered by blinking
    """
    global click_type
    global TOTAL
    mouse.click(button='right')
    click_type = 'Right Click'
    TOTAL = 0


# Factor to amplify the cursor movement by.
sclFact = 5
firstRun = True

# Declaring variables to hold the displacement
# of tracked feature in x and y direction respectively
global xC
global yC

# Setting the initial location for the cursor
mouse.move(size()[0] // 2, size()[1] // 2)


def track_nose(nose):
    """
    Function to track the tip of the nose and move the cursor accordingly
    """
    global xC
    global yC
    global firstRun
    # Finding the position of tip of nose
    cx = nose[3][0]
    cy = nose[3][1]
    if firstRun:
        xC = cx
        yC = cy
        firstRun = False
    else:
        # Calculating distance moved
        xC = cx - xC
        yC = cy - yC
        # Moving the cursor by appropriate value according to calculation
        mouse.move(mvmt_func(-xC) * sclFact, mvmt_func(yC) * sclFact, absolute=False, duration=0)
        # Resetting the current position of cursor
        xC = cx
        yC = cy


# Looping over video frames
while True:
    # Reading frames from the Video Stream
    ret, frame = vs.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detecting faces in the grayscale frame
    rects = detector(gray, 0)

    # Starting the timer to get FPS of Video Stream
    timer = cv2.getTickCount()
    # Looping  over the face detections
    for rect in rects:
        # Finding the facial landmarks for the face
        shape = predictor(gray, rect)
        # Converting the facial landmark coords to a NumPy array
        shape = shape_arr_func(shape)

        # Left eye coords
        leftEye = shape[lStart:lEnd]
        # Right eye coords
        rightEye = shape[rStart:rEnd]
        # Coords for the nose
        nose = shape[nStart:nEnd]

        # Finding E.A.R for the left and right eye
        leftEAR = ear_func(leftEye)
        rightEAR = ear_func(rightEye)

        # Finding the average for E.A.R together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # Tracking the nose
        track_nose(nose)

        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        noseHull = cv2.convexHull(nose)

        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [noseHull], -1, (0, 255, 0), 1)

        # Drawing the number of Blinks
        cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (450, 30),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        cv2.putText(frame, "Click Type: {}".format(click_type), (10, 60),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        cv2.putText(frame, "FPS: {}".format(round(fps, 2)), (10, 90),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)
        cv2.putText(frame, "Press q to exit", (10, 200),
                    cv2.FONT_HERSHEY_PLAIN, 2, (147, 20, 255), 2)

        # Increment blink counter if the EAR was less than specified threshold
        if ear < EYE_AR_THRESH:
            COUNTER += 1

        else:
            # If the eyes were closed for a sufficient number of frames
            # then increment the total number of blinks
            if EYE_AR_CONSEC_FRAMES_MIN <= COUNTER <= EYE_AR_CONSEC_FRAMES_MAX:
                TOTAL += 1
                # Giving the user a 0.7s buffer to blink
                # the required amount of times
                threading.Timer(0.7, left_click_func).start()
            # Perform a right click if the eyes were closed
            # for more than the limit for left clicks
            elif COUNTER > EYE_AR_CONSEC_FRAMES_MAX:
                TOTAL += 1
                threading.Timer(0.1, right_click_func).start()
            # Reset the COUNTER after a click event
            COUNTER = 0

    # show the frame
    cv2.imshow("Frame", frame)

    # Close the program if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()