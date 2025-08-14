# asl-interpreter (WIP)
A machine learning project to detect ASL (American Sign Language) alphabet and symbols and translate to text, using MediaPipe + Python + TensorFlow. Currently in the learning and prototyping phase: exploring hand tracking soltions with MediaPipe.

## features (planned)
- real-time hand detection using MediaPipe
- ASL alphabet recognition
- text output of detected signs
    - potential speech output

## basic setup
- python __3.12.7__
- other dependencies in `requirements.txt`

> [!NOTE]
> CURRENTLY: __detect_test.py__ is basic hand-tracking, while __gesture_test.py__ is MediaPipe's basic gesture recognition model.

Feel free to play around with these scripts:
```
git clone https://github.com/monishramj/asl-interpreter.git
cd asl-interpreter
pip install -r requirements.txt
```