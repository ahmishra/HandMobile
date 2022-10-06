from PIL import ImageFont, ImageDraw, Image  # Displaying text on screen
import mediapipe as mp  # Detecting hands
import numpy as np  # Arithmetic & Science
import serial  # To communicate with Arduino
import time  # Calculating time
import cv2  # Computer vision



class Camera:
    def __init__(self, camera, controller, hardware_enabled=False, ser=None):
        """
        Initialzing variables that are to be used through out the class
        """

        self.camera = camera
        self.mp_hands = mp.solutions.hands  # Hand detecting model
        self.mp_draw = mp.solutions.drawing_utils  # Displaying utilities
        self.finger_tip_ids = finger_tip_ids = [
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            self.mp_hands.HandLandmark.RING_FINGER_TIP,
            self.mp_hands.HandLandmark.PINKY_TIP,
        ]  # Getting the tips of each finger in the hand (4: THUMB TIP, 8: INDEX TIP, ...)
        # REFER: https://mediapipe.dev/images/mobile/hand_landmarks.png
        self.next_frame_time = 0  # Starting frame time to calculate FPS
        self.prev_frame_time = 0  # Previous frame time to calculate FPS
        self.model = self.mp_hands.Hands(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )  # Initialzing the model to track the hand
        self.hardware_enabled = hardware_enabled # Whether hardware is connected
        if self.hardware_enabled: # If hardware enabled
            self.ser = ser # Set serial to serial object
        else:
            self.ser = None # Else None
        self.controller = controller.upper() # Controlling hand


    def draw_markers(self, results, frame):
        """
        Drawing the landmarks present on the surface of the hand.
        > params
        results: The result which was obtained after running the hand model on the frame
        frame: The current frame
        """

        if (
            results.multi_hand_landmarks
        ):  # If a hand is present draw markers on the hand to track it (optional, requires DRAW to be activated)
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    image=frame,
                    landmark_list=hand_landmarks,
                    connections=self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.mp_draw.DrawingSpec(
                        color=(255, 255, 255), thickness=2, circle_radius=2
                    ),
                    connection_drawing_spec=self.mp_draw.DrawingSpec(
                        color=(33, 33, 33), thickness=2, circle_radius=2
                    ),
                )  # Drawing markers

    def detect_fingers(self, results, finger_status, count):
        """
        Detecting the fingers present in the frame.
        > params
        results: The result which was obtained after running the hand model on the frame
        finger_status: The current status of each of the fingers in the dictionary
        count: The current count of the fingers which are activated
        """

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
                self.finger_tip_ids
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
                self.mp_hands.HandLandmark.THUMB_TIP
            ].x  # Retrieving the X value for the thumb's tip
            thumb_tip_mid_point_x = hand_landmarks.landmark[
                self.mp_hands.HandLandmark.THUMB_TIP - 2
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

    def check_motion(self, count):
        """
        Determining the motion with respect to the number of fingers that are activated.
        > params
        count: The current count of the fingers which are activated
        """

        if count[self.controller] == 1:  # 1 finger in the controlling hand: Forward
            return "F"
        elif count[self.controller] == 2:  # 2 fingers in the controlling hand: Backward
            return "B"
        elif count[self.controller] == 3:  # 3 fingers in the controlling hand: Right
            return "R"
        elif count[self.controller] == 4:  # 4 fingers in the controlling hand: Left
            return "L"
        return " "

    def update_fps(self):
        """
        Updating the FPS in real time
        > params
        None
        """

        self.new_frame_time = time.time()
        fps = str(int(1 / (self.new_frame_time - self.prev_frame_time)))
        self.prev_frame_time = self.new_frame_time
        return fps

    def display_info(self, frame_pil, motion, fps, fontpath):
        """
        Displaying the miscellaneous information on the frame
        > params
        frame_pil: The PIL compatible frame
        motion: The current direction of motion
        fps: The current FPS
        fontpath: The path of the font to be used
        """

        font = ImageFont.truetype(fontpath, 32)  # Initialzing font
        draw = ImageDraw.Draw(
            frame_pil
        )  # Initialzing the drawing capabilties of text on the frame
        draw.text(
            (20, 15), f"MOTION: {motion}", font=font, fill=(255, 255, 255, 1)
        )  # Drawing text (Motion information)
        draw.text(
            (515, 15), f"FPS: {fps}", font=font, fill=(255, 255, 255, 1)
        )  # Drawing text (FPS information)

    def process_frame(self):
        """
        Many permutations are done on the frame here to get the end result
        > params
        None
        """

        frame = self.camera.read()  # Getting the frame ready
        frame = cv2.flip(frame, 1)  # Flipping the frame to avoid mirroring
        results = self.model.process(
            frame
        )  # Detecting a hand if it is present in the current frame

        self.draw_markers(results, frame)

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
            self.detect_fingers(results, finger_status, count)

        # Miscellaneous Information
        motion = self.check_motion(count)
        fps = self.update_fps()

        # Displaying
        frame_pil = Image.fromarray(
            frame
        )  # Converting CV2 frame to PIL friendly-information
        self.display_info(frame_pil, motion, fps, "Poppins-Regular.ttf")
        frame = np.array(
            frame_pil
        )  # Converting PIL friendly-information back into a CV2 frame

        if self.hardware_enabled:
            self.ser.write(motion.encode())  # Sending Motion Data

        return frame

    def mainloop(self):
        """
        Handles the entire program loop
        > params
        None
        """

        while True:
            frame = self.process_frame()
            _, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()
            yield (
                b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )

        if self.hardware_enabled:
            self.ser.close()  # Closing Serial Communications

# Aryan Mishra Copyright(C) 2022
