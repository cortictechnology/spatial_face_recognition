from pathlib import Path
import os
import time
import cv2
import numpy as np
import argparse
import depthai as dai
from utils import *

COLOR = [(0, 153, 0), (234, 187, 105), (175, 119, 212), (80, 190, 168)]

class SpatialFaceRecognizer:

    def __init__(self,
                fd_path="models/face-detection-0200.blob", 
                fd_score_thresh=0.65,
                lm_path="models/landmarks-regression-retail-0009_openvino_2021.2_6shave.blob",
                show_lm=False,
                ag_path="models/age-gender-recognition-retail-0013_openvino_2021.2_6shave.blob",
                fr_path="models/mobilefacenet.blob"):

        self.fd_path = fd_path
        self.fd_score_thresh = fd_score_thresh
        self.lm_path = lm_path
        self.show_lm = show_lm
        self.ag_path = ag_path
        self.fr_path = fr_path
        self.database_location = "./database"

        self.add_new_face = False

        self.preview_width = 455
        self.preview_height = 256

        self.ft = cv2.freetype.createFreeType2()
        self.ft.loadFontData(fontFileName='HelveticaNeue.ttf', id=0)

        self.ref_landmarks = np.array([
            [38.2946, 51.6963], 
            [73.5318, 51.5014], 
            [56.0252, 71.7366], 
            [41.5493, 92.3655], 
            [70.7299, 92.2041]], dtype=np.float32)
        self.ref_landmarks = np.expand_dims(self.ref_landmarks, axis=0)

        self.face_database = {"Names": [], "Features": None}

        self.device = dai.Device(self.create_pipeline())
        self.device.startPipeline()

        self.q_video = self.device.getOutputQueue(name="cam_out", maxSize=4, blocking=False)
        self.q_detections = self.device.getOutputQueue(name="detections", maxSize=4, blocking=False)
        self.q_lm_in = self.device.getInputQueue(name="lm_in")
        self.q_lm_out = self.device.getOutputQueue(name="lm_out", maxSize=4, blocking=False)
        self.q_ag_in = self.device.getInputQueue(name="ag_in")
        self.q_ag_out = self.device.getOutputQueue(name="ag_out", maxSize=4, blocking=False)
        self.q_fr_in = self.device.getInputQueue(name="fr_in")
        self.q_fr_out = self.device.getOutputQueue(name="fr_out", maxSize=4, blocking=False)

    
    def create_pipeline(self):
        print("Creating pipeline...")
        pipeline = dai.Pipeline()
        #pipeline.setOpenVINOVersion(version = dai.OpenVINO.Version.VERSION_2021_2)

        print("Creating Color Camera...")
        cam = pipeline.createColorCamera()
        cam.setPreviewSize(256, 256)
        cam.setPreviewKeepAspectRatio(False)
        cam.setInterleaved(False)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam_out = pipeline.createXLinkOut()
        cam_out.setStreamName("cam_out")

        print("Creating Left and Right Mono Camera...")
        monoLeft = pipeline.createMonoCamera()
        monoRight = pipeline.createMonoCamera()
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        print("Creating stereo node...")
        stereo = pipeline.createStereoDepth()
        stereo.setOutputDepth(True)
        stereo.setConfidenceThreshold(255)
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)
       
        print("Creating Spatial Face Detection Network...")
        spatialDetectionNetwork = pipeline.createMobileNetSpatialDetectionNetwork()
        spatialDetectionNetwork.setBlobPath(str(Path(self.fd_path).resolve().absolute()))
        self.fd_input_length = 256
        spatialDetectionNetwork.setConfidenceThreshold(self.fd_score_thresh)
        spatialDetectionNetwork.input.setBlocking(False)
        spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
        spatialDetectionNetwork.setDepthLowerThreshold(100)
        spatialDetectionNetwork.setDepthUpperThreshold(5000)
        detection_out = pipeline.createXLinkOut()
        detection_out.setStreamName("detections")
        spatialDetectionNetwork.out.link(detection_out.input)

        cam.preview.link(spatialDetectionNetwork.input)
        spatialDetectionNetwork.passthrough.link(cam_out.input)
        stereo.depth.link(spatialDetectionNetwork.inputDepth)

        print("Creating Face Landmark Network...")
        lm_in = pipeline.createXLinkIn()
        lm_in.setStreamName("lm_in")
        lm_nn = pipeline.createNeuralNetwork()
        lm_nn.setBlobPath(str(Path(self.lm_path).resolve().absolute()))
        self.lm_input_length = 48
        lm_out = pipeline.createXLinkOut()
        lm_out.setStreamName("lm_out")
        lm_in.out.link(lm_nn.input)
        lm_nn.out.link(lm_out.input)

        print("Creating Face Recognition Network...")
        fr_in = pipeline.createXLinkIn()
        fr_in.setStreamName("fr_in")
        fr_nn = pipeline.createNeuralNetwork()
        fr_nn.setBlobPath(str(Path(self.fr_path).resolve().absolute()))
        self.fr_input_length = 112
        fr_out = pipeline.createXLinkOut()
        fr_out.setStreamName("fr_out")
        fr_in.out.link(fr_nn.input)
        fr_nn.out.link(fr_out.input)

        print("Creating Age and Gender Network...")
        ag_in = pipeline.createXLinkIn()
        ag_in.setStreamName("ag_in")
        ag_nn = pipeline.createNeuralNetwork()
        ag_nn.setBlobPath(str(Path(self.ag_path).resolve().absolute()))
        self.ag_input_length = 62
        ag_out = pipeline.createXLinkOut()
        ag_out.setStreamName("ag_out")
        ag_in.out.link(ag_nn.input)
        ag_nn.out.link(ag_out.input)

        print("Pipeline created.")
        return pipeline

    
    def lm_render(self, img, face_landmarks):
        cv2.circle(img, tuple(face_landmarks[:2]), 2, (0, 0, 255), -1)  # Right eye
        cv2.circle(img, tuple(face_landmarks[2:4]), 2, (0, 255, 0), -1)  # Left eye
        cv2.circle(img, tuple(face_landmarks[4:6]), 2, (0, 255, 255), -1)  # Nose
        cv2.circle(img, tuple(face_landmarks[6:8]), 2, (0, 0, 255), -1)  # Right mouth
        cv2.circle(img, tuple(face_landmarks[8:]), 2, (0, 255, 0), -1)  # Left mouth


    def get_face_landmarks(self, face_frame):
        frame_lm = dai.ImgFrame()
        frame_lm.setWidth(self.lm_input_length)
        frame_lm.setHeight(self.lm_input_length)
        frame_lm.setData(to_planar(face_frame, (self.lm_input_length, self.lm_input_length)))
        self.q_lm_in.send(frame_lm)
        face_landmarks = self.q_lm_out.get().getFirstLayerFp16()
        face_landmarks = frame_norm(face_frame, face_landmarks)
        return face_landmarks


    def get_face_features(self, aligned_face):
        frame_fr = dai.ImgFrame()
        frame_fr.setWidth(self.fr_input_length)
        frame_fr.setHeight(self.fr_input_length)
        frame_fr.setData(to_planar(aligned_face, (self.fr_input_length, self.fr_input_length)))
        self.q_fr_in.send(frame_fr)
        face_features = np.array(self.q_fr_out.get().getFirstLayerFp16()).astype(np.float32)
        face_features_norm = np.linalg.norm(face_features)
        face_features = face_features / face_features_norm
        return face_features


    def load_database(self, database_location):
        for dir in os.listdir(database_location):
            item = os.path.join(database_location, dir)
            if os.path.isdir(item):
                for file in os.listdir(item):
                    if file.endswith(".bin"):
                        try:
                            feature = np.fromfile(item+"/features.bin", np.float32)
                            self.face_database["Names"].append(dir)
                            if self.face_database["Features"] is None:
                                self.face_database["Features"] = feature
                            else:
                                self.face_database["Features"] = np.vstack((self.face_database["Features"], feature))
                        except:
                            continue


    def get_best_match_identity(self, similarity_scores, threshold=0.81):
        sort_idx = np.argsort(-similarity_scores)
        #print(sort_idx)
        if similarity_scores[sort_idx[0]] >= threshold:
            return self.face_database["Names"][sort_idx[0]]
        else:
            return "Unknown"


    def get_age_gender(self, aligned_face):
        frame_ag = dai.ImgFrame()
        frame_ag.setWidth(self.ag_input_length)
        frame_ag.setHeight(self.ag_input_length)
        frame_ag.setData(to_planar(aligned_face, (self.ag_input_length, self.ag_input_length)))
        self.q_ag_in.send(frame_ag)
        age_gender_result = self.q_ag_out.get()
        age = int(float(np.squeeze(np.array(age_gender_result.getLayerFp16('age_conv3')))) * 100)
        gender = np.squeeze(np.array(age_gender_result.getLayerFp16('prob')))
        gender_str = "female" if gender[0] > gender[1] else "male"
        return age, gender_str


    def draw_disconnected_rect(self, img, pt1, pt2, color, thickness):
        width = pt2[0] - pt1[0]
        height = pt2[1] - pt1[1]
        cv2.line(img, pt1, (pt1[0] + width // 4, pt1[1]), color, thickness)
        cv2.line(img, pt1, (pt1[0], pt1[1] + height // 4), color, thickness)
        cv2.line(img, (pt2[0] - width // 4, pt1[1]), (pt2[0], pt1[1]), color, thickness)
        cv2.line(img, (pt2[0], pt1[1]), (pt2[0], pt1[1] + height // 4), color, thickness)
        cv2.line(img, (pt1[0], pt2[1]), (pt1[0] + width // 4, pt2[1]), color, thickness)
        cv2.line(img, (pt1[0], pt2[1] - height // 4), (pt1[0], pt2[1]), color, thickness)
        cv2.line(img, pt2, (pt2[0] - width // 4, pt2[1]), color, thickness)
        cv2.line(img, (pt2[0], pt2[1] - height // 4), pt2, color, thickness)


    def run(self):
        if self.database_location != "":
            self.load_database(self.database_location)
        while True:
            in_video = self.q_video.get()
            video_frame = in_video.getCvFrame()

            video_frame = cv2.resize(video_frame, (self.preview_width, self.preview_height))
            annotated_frame = video_frame.copy()

            detected_faces = self.q_detections.get().detections

            height = annotated_frame.shape[0]
            width  = annotated_frame.shape[1]

            for detection in detected_faces:
                x1 = int(detection.xmin * width)
                x2 = int(detection.xmax * width)
                y1 = int(detection.ymin * height)
                y2 = int(detection.ymax * height)

                if x1 < 0:
                    x1 = 0
                if y1 < 0:
                    y1 = 0
                if x2 >= width:
                    x2 = width - 1
                if y2 >= height:
                    y2 = height - 1

                x_center = int(x1 + (x2 - x1) / 2)

                z_text = f"Distance: {int(detection.spatialCoordinates.z)} mm"

                textSize = self.ft.getTextSize(z_text, fontHeight=14, thickness=-1)[0]

                face_frame = video_frame[y1:y2, x1:x2]

                face_landmarks = self.get_face_landmarks(face_frame)
                for i in range(5):
                    face_landmarks[i * 2] = face_landmarks[i * 2] + x1
                    face_landmarks[i * 2 + 1] = face_landmarks[i * 2 + 1] + y1

                draw_landmarks = face_landmarks

                face_landmarks = face_landmarks.reshape((-1, 2))

                person_name = "Unknown"
                aligned_face = norm_crop(video_frame, face_landmarks, self.ref_landmarks)
                face_features = self.get_face_features(aligned_face)

                if len(self.face_database["Names"]) > 0:
                    similarity = np.dot(self.face_database["Features"], face_features.T).squeeze()
                    if not isinstance(similarity, np.ndarray):
                        similarity = np.array([similarity])
                    # Perform some nolinear scaling to the similarity
                    similarity = 1.0 / (1 + np.exp(-1 * (similarity - 0.38) * 10))
                    person_name = self.get_best_match_identity(similarity)
                
                if person_name == "Unknown":
                    if self.show_lm:
                        self.lm_render(annotated_frame, draw_landmarks)
                    age, gender = self.get_age_gender(aligned_face)
                    age_gender_text = str(age) + " years old " + gender
                    textSize_age_gender = self.ft.getTextSize(age_gender_text, fontHeight=14, thickness=-1)[0]
                    textSize_distance = self.ft.getTextSize(z_text, fontHeight=14, thickness=-1)[0]
                    text_width = max(textSize_age_gender[0], textSize_distance[0])
                    cv2.rectangle(annotated_frame, (x_center - text_width // 2 - 5, y1 - 22), (x_center - text_width // 2 + text_width + 10, y1 - 39), COLOR[0], -1)
                    self.ft.putText(img=annotated_frame, text=age_gender_text , org=(x_center - text_width // 2, y1 - 25), fontHeight=14, color=(255, 255, 255), thickness=-1, line_type=cv2.LINE_AA, bottomLeftOrigin=True)
                    cv2.rectangle(annotated_frame, (x_center - text_width // 2 - 5, y1 - 5), (x_center - text_width // 2 + text_width + 10, y1 - 22), COLOR[0], -1)
                    self.ft.putText(img=annotated_frame, text=z_text, org=(x_center - text_width // 2, y1 - 8), fontHeight=14, color=(255, 255, 255), thickness=-1, line_type=cv2.LINE_AA, bottomLeftOrigin=True)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), COLOR[0], cv2.FONT_HERSHEY_SIMPLEX)                        
                else:
                    self.draw_disconnected_rect(annotated_frame, (x1, y1), (x2, y2), COLOR[1], 1)
                    name_text = "Name: " + person_name[0:person_name.find('-')]
                    textSize = self.ft.getTextSize(name_text, fontHeight=14, thickness=-1)[0]
                    cv2.rectangle(annotated_frame, (x_center - textSize[0] // 2 - 5, y1 - 5), (x_center - textSize[0] // 2 + textSize[0] + 5, y1 - 22), COLOR[1], -1)
                    self.ft.putText(img=annotated_frame, text=name_text , org=(x_center - textSize[0] // 2, y1 - 8), fontHeight=14, color=(255, 255, 255), thickness=-1, line_type=cv2.LINE_AA, bottomLeftOrigin=True)
                    
                if person_name == "Unknown" and self.add_new_face:
                    if self.database_location != "":
                        name = input("Please enter your name: ")
                        os.mkdir(self.database_location + "/" + name)
                        cv2.imwrite(self.database_location + name + "/photo.jpg", video_frame)
                        face_features.tofile(self.database_location + name + "/features.bin")
                        self.face_database["Names"].append(name)
                        if self.face_database["Features"] is None:
                            self.face_database["Features"] = face_features
                        else:
                            self.face_database["Features"] = np.vstack((self.face_database["Features"], face_features))
                    else:
                        print("Run the program with a database location first!")
                    self.add_new_face = False
                
            cv2.imshow("Spatial Face Recognition", annotated_frame)

            key = cv2.waitKey(1) 
            if key == ord('q'):
                break
            elif key == ord('a'):
                self.add_new_face = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fd_m", default="models/face-detection-0200.blob", type=str,
                        help="Path to a blob file for face detection model (default=%(default)s)")
    parser.add_argument("--lm_m", default="models/landmarks-regression-retail-0009_openvino_2021.2_6shave.blob", type=str,
                        help="Path to a blob file for landmark model (default=%(default)s)")
    parser.add_argument('--show_lm', action="store_true", 
                        help="Show the face landmarks on image")
    parser.add_argument("--ag_m", default="models/age-gender-recognition-retail-0013_openvino_2021.2_6shave.blob", type=str,
                        help="Path to a blob file for age and gender estimation model (default=%(default)s)")
    parser.add_argument("--fr_m", default="models/mobilefacenet.blob", type=str,
                        help="Path to a blob file for face recognition model (default=%(default)s)")
    args = parser.parse_args()

    sf = SpatialFaceRecognizer(fd_path=args.fd_m, lm_path=args.lm_m, show_lm=args.show_lm, ag_path=args.ag_m, fr_path=args.fr_m)
    sf.run()
