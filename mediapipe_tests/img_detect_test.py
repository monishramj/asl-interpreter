import mediapipe as mp
import pandas as pd

model_path = 'mediapipe_models/hand_landmarker.task'
vision = mp.tasks.vision

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options = BaseOptions(model_asset_path=model_path), 
    running_mode = VisionRunningMode.IMAGE, 
    num_hands = 1, 
    min_tracking_confidence = .7, 
    min_hand_detection_confidence = .6
)

mp_image = mp.Image.create_from_file('test_img.jpg')

with HandLandmarker.create_from_options(options) as landmarker:

    result = landmarker.detect(mp_image)

    print('hand landmarker result: {}'.format(result) + '\n')
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
        df = pd.DataFrame([landmarks_data])
        print(df)

