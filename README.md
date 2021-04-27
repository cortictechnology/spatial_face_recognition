# Face Detection, face recognition, age and gender estimation with spatial information

This is the source code of the article: [Importance of Accurate Facial Landmarks](https://www.cortic.ca/post/importance-of-accurate-facial-landmarks)

## Install dependencies

On a Raspberry Pi 4B or a PC with Ubuntu/Debian, run in terminal:

```
git clone https://github.com/cortictechnology/spatial_face_recognition.git
cd spatial_face_recognition
bash install_dependencies.sh
```

## To run

1. Make sure the OAK-D device is plug into the Pi or PC.
2. In the terminal, run
```
python3 main.py
```

You can enable the drawing of face landmarks by adding in the --show_lm flag:
```
python3 main.py --show_lm
```
We have included some sample celebrities in the database, you can follow the next section to add your own face into the database.

## To add your face into the database

1. When the program is running, look into the camera with your front face.
2. Press the "a" key on the keyboard.
3. In the terminal, enter your name and press Enter.
4. Now move your head around to capture different angles of your face.
5. You should see yourself being recognized.

## Model description

In the models folder, 4 models are provided:

1. face-detection-0200.blob: Face detection nework from [OpenVino's website](https://docs.openvinotoolkit.org/latest/omz_models_model_face_detection_0200.html)
2. landmarks-regression-retail-0009_openvino_2021.2_6shave.blob: Face landmark detction model from [depthAI's sample project](https://github.com/luxonis/depthai-experiments/tree/master/gen2-nn-sync)
3. age-gender-recognition-retail-0013_openvino_2021.2_6shave.blob: Age and Gender estimation model from [depthAI's sample project](https://github.com/luxonis/depthai-experiments/tree/master/gen2-age-gender)
4. mobilefacenet.blob: Face recognition model converted from [insightface's mxnet model](https://github.com/deepinsight/insightface/wiki/Model-Zoo)

