import argparse
import sys
import time

import threading
import asyncio

import websockets

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Global variables to calculate FPS
COUNTER, FPS = 0, 0
START_TIME = time.time()
DETECTION_RESULT = None
region_width = 300
region_height = 300
r_touch = False
l_touch = False
ix = -1
iy = -1

zoom_in = False
zoom_out = False

dist = 0

def crop_top_right(image, width, height):
    # Calculate the coordinates of the top right corner
    start_row, start_col = 0, image.shape[1] - width
    end_row, end_col = start_row + height, start_col + width
    # Crop the image
    img_cropped = image[start_row:end_row, start_col:end_col]
    return img_cropped

def get_dist(landmark1: NormalizedLandmark, landmark2: NormalizedLandmark):
    x1 = landmark1.x
    y1 = landmark1.y

    x2 = landmark2.x
    y2 = landmark2.y

    x = x2 - x1
    y = y2 - y1

    dist = (x**2+y**2)**0.5
    return dist

def run(model: str, num_hands: int,
        min_hand_detection_confidence: float,
        min_hand_presence_confidence: float, min_tracking_confidence: float,
        camera_id: int, width: int, height: int,
        region_width: int, region_height: int,
        headless: int, debug: int) -> None:

    """
    Continuously run inference on images acquired from the camera.

    Args:
        model: Name of the hand landmarker model bundle.
        num_hands: Max number of hands that can be detected by the landmarker.
        min_hand_detection_confidence: The minimum confidence score for hand
            detection to be considered successful.
        min_hand_presence_confidence: The minimum confidence score of hand
            presence score in the hand landmark detection.
        min_tracking_confidence: The minimum confidence score for the hand
            tracking to be considered successful.
        camera_id: The camera id to be passed to OpenCV.
        width: The width of the frame captured from the camera.
        height: The height of the frame captured from the camera.
        headless: The flag to run the script without cam feed.
        debug: The flag to print the handedness and landmarks.
    """
    global zoom_out, zoom_in, r_touch, l_touch, ix, iy

    # Start capturing video input from the camera
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Visualization parameters
    row_size = 50  # pixels
    left_margin = 24  # pixels
    text_color = (0, 0, 0)  # black
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10

    def save_result(result: vision.HandLandmarkerResult,
                    unused_output_image: mp.Image, timestamp_ms: int):
        global FPS, COUNTER, START_TIME, DETECTION_RESULT

        # Calculate the FPS
        if COUNTER % fps_avg_frame_count == 0:
            FPS = fps_avg_frame_count / (time.time() - START_TIME)
            START_TIME = time.time()

        DETECTION_RESULT = result
        COUNTER += 1

    # Initialize the hand landmarker model
    base_options = python.BaseOptions(model_asset_path=model)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=num_hands,
        min_hand_detection_confidence=min_hand_detection_confidence,
        min_hand_presence_confidence=min_hand_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        result_callback=save_result)
    detector = vision.HandLandmarker.create_from_options(options)

    # try:

    # Continuously capture images from the camera and run inference
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit(
                'ERROR: Unable to read from webcam. Please verify your webcam settings.'
            )

        image = cv2.flip(image, 1)
        image = crop_top_right(image, region_width, region_height)

        # Convert the image from BGR to RGB as required by the TFLite model.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Run hand landmarker using the model.
        detector.detect_async(mp_image, time.time_ns() // 1_000_000)

        fps_text = 'FPS = {:.1f}'.format(FPS)
        if headless or debug:
            print(fps_text)

        if not headless:
            # Show the FPS
            text_location = (left_margin, row_size)
            current_frame = image
            cv2.putText(current_frame, fps_text, text_location,
                        cv2.FONT_HERSHEY_DUPLEX,
                        font_size, text_color, font_thickness, cv2.LINE_AA)

        # Landmark visualization parameters.
        MARGIN = 10  # pixels
        FONT_SIZE = 1
        FONT_THICKNESS = 1
        HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green

        r_touch = False
        l_touch = False
        ix = -1
        iy = -1
        zoom_in = False
        zoom_out = False

        if DETECTION_RESULT:
            if len(DETECTION_RESULT.hand_landmarks) == 2:
                h0_index_tip = DETECTION_RESULT.hand_landmarks[0][8]
                h0_handedness = DETECTION_RESULT.handedness[0][0].category_name
                
                h1_index_tip = DETECTION_RESULT.hand_landmarks[1][8]
                h1_handedness = DETECTION_RESULT.handedness[1][0].category_name

                if h0_handedness != h1_handedness:
                    dist = get_dist(h0_index_tip, h1_index_tip) - dist
                    if dist<-0.3:
                        zoom_in = True
                    elif dist>0.3:
                        zoom_out = True
                else:
                    dist = 0

            # Draw landmarks and indicate handedness.
            for idx in range(len(DETECTION_RESULT.hand_landmarks)):
                hand_landmarks = DETECTION_RESULT.hand_landmarks[idx]
                handedness = DETECTION_RESULT.handedness[idx]
                l_touch_check = get_dist(hand_landmarks[8], hand_landmarks[4])
                r_touch_check = get_dist(hand_landmarks[12], hand_landmarks[4])

                if handedness[0].category_name == "Left":
                    ix = hand_landmarks[5].x
                    iy = hand_landmarks[5].y

                # print(round(arctan(index_vert), 2))

                # print(touch)
                if l_touch_check<0.03:
                    # print("\033[91m Right Touch Detected \033[0m")
                    l_touch = True

                if r_touch_check<0.03:
                    # print("\033[91m Left Touch Detected \033[0m")
                    r_touch = True

                # Draw the hand landmarks
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y,
                                                    z=landmark.z) for landmark
                    in hand_landmarks
                ])

                if debug:
                    print(handedness, hand_landmarks_proto)

                if not headless:
                    mp_drawing.draw_landmarks(
                        current_frame,
                        hand_landmarks_proto,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                    # Get the top left corner of the detected hand's bounding box.
                    height, width, _ = current_frame.shape
                    x_coordinates = [landmark.x for landmark in hand_landmarks]
                    y_coordinates = [landmark.y for landmark in hand_landmarks]
                    text_x = int(min(x_coordinates) * width)
                    text_y = int(min(y_coordinates) * height) - MARGIN

                    # Draw handedness (left or right hand) on the image.
                    cv2.putText(current_frame, f"{handedness[0].category_name}",
                                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS,
                                cv2.LINE_AA)
        if not headless:
            cv2.imshow('hand_landmarker', current_frame)

        # Stop the program if the ESC key is pressed.
        if cv2.waitKey(1) == 27:
            break
    # except KeyboardInterrupt:
    #     print("\033[91m Closing program...\033[0m")

    # finally:
    #     detector.close()
    #     cap.release()
    #     cv2.destroyAllWindows()
    #     sys.exit(1)

    detector.close()
    cap.release()
    cv2.destroyAllWindows()

async def send_message(ip):
    uri = f"ws://{ip}:8765"
    async with websockets.connect(uri) as websocket:
        while True:
            message = f"{ix},{iy},"

            if r_touch:
                message += "r_touch,"
                print("\033[91m Right Touch Detected \033[0m")

            if l_touch:
                message += "l_touch,"
                print("\033[91m Left Touch Detected \033[0m")
            
            if zoom_in:
                message += "zoom_in,"
                print("\033[91m Zoom In Detected \033[0m")

            if zoom_out:
                message += "zoom_out,"
                print("\033[91m Zoom Out Detected \033[0m")

            await websocket.send(message)

            # Receive and print the result from the server
            result = await websocket.recv()
            print(f"Server response: {result}")

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model',
        help='Name of the hand landmarker model bundle.',
        required=False,
        default='hand_landmarker.task')
    parser.add_argument(
        '--numHands',
        help='Max number of hands that can be detected by the landmarker.',
        required=False,
        default=2)
    parser.add_argument(
        '--minHandDetectionConfidence',
        help='The minimum confidence score for hand detection to be considered '
             'successful.',
        required=False,
        default=0.5)
    parser.add_argument(
        '--minHandPresenceConfidence',
        help='The minimum confidence score of hand presence score in the hand '
             'landmark detection.',
        required=False,
        default=0.5)
    parser.add_argument(
        '--minTrackingConfidence',
        help='The minimum confidence score for the hand tracking to be '
             'considered successful.',
        required=False,
        default=0.5)

    parser.add_argument(
        '--cameraId', help='Id of camera.', required=False, default=0)
    parser.add_argument(
        '--frameWidth',
        help='Width of frame to capture from camera.',
        required=False,
        default=1280)
    parser.add_argument(
        '--frameHeight',
        help='Height of frame to capture from camera.',
        required=False,
        default=960)
    parser.add_argument(
        '--workWidth',
        help='',
        required=False,
        default=300)
    parser.add_argument(
        '--workHeight',
        help='Print the handedness and landmarks.',
        required=False,
        default=300)
    parser.add_argument(
        '--headless',
        help='Run the script without cam feed.',
        required=False,
        default=0)
    parser.add_argument(
        '--debug',
        help='Print the handedness and landmarks.',
        required=False,
        default=0)
    
    parser.add_argument(
        '--IP',
        help='IP address of the server',
        required=False,
        default="localhost")

    args = parser.parse_args()

    IP = args.IP
    print(IP)

    inference_thread = threading.Thread(target=run, args=(
        args.model, int(args.numHands), args.minHandDetectionConfidence,
        args.minHandPresenceConfidence, args.minTrackingConfidence,
        int(args.cameraId), args.frameWidth, args.frameHeight,
        int(args.workWidth), int(args.workHeight),
        args.headless, args.debug)
    )

    inference_thread.start()
    asyncio.run(send_message(IP))

    # inference_thread.join()
    # client_thread.join()

if __name__ == '__main__':
    main()