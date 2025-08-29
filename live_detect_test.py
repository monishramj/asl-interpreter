import mediapipe as mp
import pandas as pd
import cv2 as cv 
import time

model_path = 'hand_landmarker.task'
vision = mp.tasks.vision

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
HandLandmarkerResult = vision.HandLandmarkerResult
VisionRunningMode = vision.RunningMode

frame_result = None
lastest_landmarks = None

def view_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int): # type: ignore
    global frame_result 
    frame_result = result
    # print('hand landmarker result: {}'.format(result) + '\n')

    if result.hand_landmarks and len(result.hand_landmarks) > 0:
        landmarks_data = {}
        landmarks = result.hand_landmarks[0]

        for i, lm in enumerate(landmarks):
            x = lm.x
            landmarks_data[f'x{i}'] = x

            y = lm.y
            landmarks_data[f'y{i}'] = y

            z = lm.z
            landmarks_data[f'z{i}'] = z

        lastest_landmarks = landmarks_data


    # df = pd.DataFrame([landmarks_data])
    
    # print(df)
                

                
    return


def draw_pts(result, img):
    if result != None:
        h, w, _ = img.shape
        marklist = result.hand_landmarks 
        for hand in marklist:
            for landmark in hand:
                x = int(landmark.x * w); y = int(landmark.y * h)
                cv.circle(img, (x, y), 2, (0, 255, 255), 2)
    return

options = HandLandmarkerOptions(

    base_options = BaseOptions(model_asset_path=model_path), 
    running_mode = VisionRunningMode.LIVE_STREAM, 
    result_callback=view_result, 
    num_hands = 1, 
    min_tracking_confidence = .7, 
    min_hand_detection_confidence = .6

)

vid = cv.VideoCapture(0)

WIDTH = 1000
HEIGHT = 480

vid.set(cv.CAP_PROP_FRAME_WIDTH, WIDTH)
vid.set(cv.CAP_PROP_FRAME_HEIGHT, HEIGHT)

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = vid.read()
        frame = cv.flip(frame, 1)
        
        if not ret:
            print("Failed write frame.")
            break

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
   
        ts_ms = int(time.time() * 1000) #manual timestamp in ms

        landmarker.detect_async(mp_image, ts_ms)
        
        key = cv.waitKey(25) & 0xFF

        if key == ord('a') and lastest_landmarks:
            df = pd.DataFrame([lastest_landmarks])
            df["label"] = "A"
            df.to_csv("asl_data.csv", mode="a", header=False, index=False)
            print("Saved one sample for A")

        draw_pts(frame_result, frame)

        cv.imshow('frame', frame)
        if cv.waitKey(25) == ord('q'):
            break

vid.release()
cv.destroyAllWindows()