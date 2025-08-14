import mediapipe as mp
import cv2 as cv 
import time

model_path = 'gesture_recognizer.task'
vision = mp.tasks.vision

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = vision.GestureRecognizer
GestureRecognizerOptions = vision.GestureRecognizerOptions
GestureRecognizerResult = vision.GestureRecognizerResult
VisionRunningMode = vision.RunningMode

frame_result = None
hand_pos = "."

def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int): # type: ignore
    global frame_result, hand_pos
    frame_result = result

    if result.gestures and result.gestures[0]:
        hand_pos = result.gestures[0][0].category_name
        # print(hand_pos)
    else:
        hand_pos="."
        # print("No gesture detected.")

    # print('gesture recognition result: {}'.format(result))

def draw_pts(result, img):
    if result != None:
        h, w, _ = img.shape
        marklist = result.hand_landmarks 
        for hand in marklist:
            for landmark in hand:
                x = int(landmark.x * w); y = int(landmark.y * h)
                cv.circle(img, (x, y), 2, (0, 255, 255), 2)
    return

options = GestureRecognizerOptions(base_options = BaseOptions(model_asset_path=model_path), running_mode = VisionRunningMode.LIVE_STREAM, result_callback=print_result)

vid = cv.VideoCapture(0)

WIDTH = 480
HEIGHT = 480
font = cv.FONT_HERSHEY_SIMPLEX

vid.set(cv.CAP_PROP_FRAME_WIDTH, WIDTH)
vid.set(cv.CAP_PROP_FRAME_HEIGHT, HEIGHT)

with GestureRecognizer.create_from_options(options) as recog:
    while True:
        ret, frame = vid.read()
        frame = cv.flip(frame, 1)
        
        if not ret:
            print("Failed write frame.")
            break

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        ts_ms = int(time.time() * 1000) #manual timestamp in ms

        recog.recognize_async(mp_image, ts_ms)
        draw_pts(frame_result, frame)

        frame = cv.putText(frame, hand_pos, (50, 50), font, 1, (255,0,0), 2, cv.LINE_AA)
        
        cv.imshow('frame', frame)

        if cv.waitKey(25) == ord('q'):
            break

vid.release()
cv.destroyAllWindows()