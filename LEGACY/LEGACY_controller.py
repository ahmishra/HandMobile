CONTROLLER = "RIGHT"
DRAW = True

if DRAW:
    b, g, r, a = (
        33,  # Blue
        33,  # Green
        33,  # Red
        1,  # Alpha
    )  # If information is to be displayed, color of the text is to be provided here

# Variables required to calculate Frames Per Second
next_frame_time = 0
prev_frame_time = 0

from PIL import ImageFont, ImageDraw, Image  # Displaying text on screen
import mediapipe as mp  # Detecting hands
import numpy as np  # Arithmetic & Science
import time  # Calculating time
import cv2  # Computer Vision
from imutils.video import VideoStream

mp_draw = mp.solutions.drawing_utils  # Displaying Utilities
mp_hands = mp.solutions.hands  # Hand detecting model


finger_tip_ids = [
    mp_hands.HandLandmark.INDEX_FINGER_TIP,
    mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
    mp_hands.HandLandmark.RING_FINGER_TIP,
    mp_hands.HandLandmark.PINKY_TIP,
]  # Getting the tips of each finger in the hand (4: THUMB TIP, 8: INDEX TIP, ...)

# REFER: https://mediapipe.dev/images/mobile/hand_landmarks.png


hands = mp_hands.Hands(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)  # Intializing the model to track the hand


cap = VideoStream(usePiCamera=False).start()

while True:  # Program loop
    frame = cap.read()  # Getting the frame ready
    frame = cv2.flip(frame, 1)  # Flipping the frame to avoid mirroring


    results = hands.process(
        frame
    )  # Detecting a hand if it is present in the current frame


    if (
        results.multi_hand_landmarks and DRAW
    ):  # If a hand is present draw markers on the hand to track it (optional, requires DRAW to be activated)
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                image=frame,
                landmark_list=hand_landmarks,
                connections=mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_draw.DrawingSpec(
                    color=(255, 255, 255), thickness=2, circle_radius=2
                ),
                connection_drawing_spec=mp_draw.DrawingSpec(
                    color=(33, 33, 33), thickness=2, circle_radius=2
                ),
            )  # Drawing markers

    count = {
        "RIGHT": 0,
        "LEFT": 0,
    }  # Keeping a count of activated fingers in both hands

    finger_status = {
        "RIGHT_THUMB": False,
        "RIGHT_INDEX": False,
        "RIGHT_MIDDLE": False,
        "RIGHT_RING": False,
        "RIGHT_PINKY": False,
        "LEFT_THUMB": False,
        "LEFT_INDEX": False,
        "LEFT_MIDDLE": False,
        "LEFT_RING": False,
        "LEFT_PINKY": False,
    }  # Current status of each finger

    # Handling Logic
    if results.multi_handedness:  # If hands are present in the frame
        for hand_id, hand_info in enumerate(
            results.multi_handedness
        ):  # Get the index of the hand (0 or 1) and other miscellaneous info about the hand
            hand_label = hand_info.classification[
                0
            ].label  # Getting the predicted hand label (either left or right)
            hand_landmarks = results.multi_hand_landmarks[
                hand_id
            ]  # Getting all of the different landmarks present on the hand

            # Finger Logic
            for (
                tip_index
            ) in (
                finger_tip_ids
            ):  # Getting all fingers (except thumb, requires different logic)
                finger_name = tip_index.name.split("_")[
                    0
                ]  # Getting the finger name to index the landmarks from
                if (
                    hand_landmarks.landmark[tip_index].y
                    < hand_landmarks.landmark[tip_index - 2].y
                ):  # If the Y value of the tip of the finger is lesser than the Y value of the mid point of the finger than means the finger is raised
                    finger_status[
                        f"{hand_label.upper()}_{finger_name.upper()}"
                    ] = True  # Set the status of that specific finger of that particular hand to activated
                    if (
                        count[f"{hand_label.upper()}"] < 5
                    ):  # Not allowing the count to go above 5, if any errors arise.
                        count[
                            f"{hand_label.upper()}"
                        ] += 1  # Adding the finger count to that specific hand

        # Thumb Logic
        thumb_tip_x = hand_landmarks.landmark[
            mp_hands.HandLandmark.THUMB_TIP
        ].x  # Retrieving the X value for the thumb's tip
        thumb_tip_mid_point_x = hand_landmarks.landmark[
            mp_hands.HandLandmark.THUMB_TIP - 2
        ].x  # Retrieving the X value for the thumb's mid point

        # Different logic on both sides of the hands
        # Right Hand: If the X value of the thumb's tip is lesser than the X value of the mid point of the thumb, the thumb is raised
        # Left Hand: If the X value of the thumb's tip is greater than the X value of the mid point of the thumb, the thumb is raised

        if (hand_label == "Right" and (thumb_tip_x < thumb_tip_mid_point_x)) or (
            hand_label == "Left" and (thumb_tip_x > thumb_tip_mid_point_x)
        ):
            finger_status[
                f"{hand_label.upper()}_THUMB"
            ] = True  # Setting the particular hand's thumb to activated
            if (
                count[f"{hand_label.upper()}"] < 5
            ):  # Not allowing the count to go above 5
                count[
                    f"{hand_label.upper()}"
                ] += 1  # Adding the thumb count to that specific hand


    motion = " "  # Determining motion
    if count[CONTROLLER] == 1:  # 1 finger in the controlling hand: Forward
        motion = "F"
    elif count[CONTROLLER] == 2:  # 2 fingers in the controlling hand: Backward
        motion = "B"
    elif count[CONTROLLER] == 3:  # 3 fingers in the controlling hand: Right
        motion = "R"
    elif count[CONTROLLER] == 4:  # 4 fingers in the controlling hand: Left
        motion = "L"


    # Calculating FPS
    new_frame_time = time.time()
    fps = str(int(1 / (new_frame_time - prev_frame_time)))
    prev_frame_time = new_frame_time

    if DRAW:  # To display extra informatio, (optional, requires DRAW to be activated)
        fontpath = "static/Poppins-Regular.ttf"  # Font path
        font = ImageFont.truetype(fontpath, 32)  # Intializing font
        frame_pil = Image.fromarray(
            frame
        )  # Converting CV2 frame to PIL friendly-information
        draw = ImageDraw.Draw(
            frame_pil
        )  # Intialzing the drawing capabilties of text on the frame
        draw.text(
            (5, 5), f"Motion: {motion}", font=font, fill=(b, g, r, a)
        )  # Drawing text (Motion information)
        draw.text(
            (5, 35), f"FPS: {fps}", font=font, fill=(b, g, r, a)
        )  # Drawing text (FPS information)
        frame = np.array(
            frame_pil
        )  # Converting PIL friendly-information back into a CV2 frame

    cv2.imshow(
        "Controller", frame
    )  # Displaying the final frame after many permutations and calculations


    # Handling exit logic
    q = cv2.waitKey(1)  # Detecting for key presses
    if q == ord("q"):  # If the key pressed is "q", break the loop and exit the program
        break
        # Break
        exit()  # Quit / Exit

# Aryan Mishra Copyright(C) 2022
