from pathlib import Path
import os
import time
import cv2
import numpy as np
import argparse
import depthai as dai
from utils import *

print("Creating pipeline...")
pipeline = dai.Pipeline()

print("Creating 2D Face Detection Network...")
fd_in = pipeline.createXLinkIn()
fd_in.setStreamName("fd_in")
fd_nn = pipeline.createNeuralNetwork()
fd_nn.setBlobPath(str(Path("models/face-detection-0200.blob").resolve().absolute()))
fd_input_length = 256
fd_out = pipeline.createXLinkOut()
fd_out.setStreamName("fd_out")
fd_in.out.link(fd_nn.input)
fd_nn.out.link(fd_out.input)

print("Creating Face Landmark Network...")
lm_in = pipeline.createXLinkIn()
lm_in.setStreamName("lm_in")
lm_nn = pipeline.createNeuralNetwork()
lm_nn.setBlobPath(str(Path("models/landmarks-regression-retail-0009_openvino_2021.2_6shave.blob").resolve().absolute()))
lm_input_length = 48
lm_out = pipeline.createXLinkOut()
lm_out.setStreamName("lm_out")
lm_in.out.link(lm_nn.input)
lm_nn.out.link(lm_out.input)

print("Creating Face Recognition Network...")
fr_in = pipeline.createXLinkIn()
fr_in.setStreamName("fr_in")
fr_nn = pipeline.createNeuralNetwork()
fr_nn.setBlobPath(str(Path("models/mobilefacenet.blob").resolve().absolute()))
fr_input_length = 112
fr_out = pipeline.createXLinkOut()
fr_out.setStreamName("fr_out")
fr_in.out.link(fr_nn.input)
fr_nn.out.link(fr_out.input)

ref_landmarks = np.array([
    [38.2946, 51.6963], 
    [73.5318, 51.5014], 
    [56.0252, 71.7366], 
    [41.5493, 92.3655], 
    [70.7299, 92.2041]], dtype=np.float32)
ref_landmarks = np.expand_dims(ref_landmarks, axis=0)

device = dai.Device(pipeline)
device.startPipeline()

q_fd_in = device.getInputQueue(name="fd_in")
q_fd_out = device.getOutputQueue(name="fd_out", maxSize=4, blocking=False)

q_lm_in = device.getInputQueue(name="lm_in")
q_lm_out = device.getOutputQueue(name="lm_out", maxSize=4, blocking=False)

q_fr_in = device.getInputQueue(name="fr_in")
q_fr_out = device.getOutputQueue(name="fr_out", maxSize=4, blocking=False)

def detect_face_2d(img):
    frame_fd = dai.ImgFrame()
    frame_fd.setWidth(fd_input_length)
    frame_fd.setHeight(fd_input_length)
    frame_fd.setData(to_planar(img, (fd_input_length, fd_input_length)))
    q_fd_in.send(frame_fd)
    bboxes = np.array(q_fd_out.get().getFirstLayerFp16())
    bboxes = bboxes[:np.where(bboxes == -1)[0][0]]
    bboxes = bboxes.reshape((bboxes.size // 7, 7))
    bboxes = bboxes[bboxes[:, 2] > 0.5][:, 3:7]
    # For 2d face detection, we only return the largest face
    largest_bbox = None
    largest_area = 0
    for raw_bbox in bboxes:
        face_width = raw_bbox[2] - raw_bbox[0]
        face_height = raw_bbox[3] - raw_bbox[1]
        area = face_width * face_height
        if area > largest_area:
            largest_area = area
            largest_bbox = raw_bbox
    return largest_bbox

def get_face_landmarks(face_frame):
    frame_lm = dai.ImgFrame()
    frame_lm.setWidth(lm_input_length)
    frame_lm.setHeight(lm_input_length)
    frame_lm.setData(to_planar(face_frame, (lm_input_length, lm_input_length)))
    q_lm_in.send(frame_lm)
    face_landmarks = q_lm_out.get().getFirstLayerFp16()
    face_landmarks = frame_norm(face_frame, face_landmarks)
    return face_landmarks

def get_face_features(aligned_face):
    frame_fr = dai.ImgFrame()
    frame_fr.setWidth(fr_input_length)
    frame_fr.setHeight(fr_input_length)
    frame_fr.setData(to_planar(aligned_face, (fr_input_length, fr_input_length)))
    q_fr_in.send(frame_fr)
    face_features = np.array(q_fr_out.get().getFirstLayerFp16()).astype(np.float32)
    face_features_norm = np.linalg.norm(face_features)
    face_features = face_features / face_features_norm
    return face_features

def generate_database(image_location):
    for dir in os.listdir(image_location):
        item = os.path.join(image_location, dir)
        if os.path.isdir(item):
            count = 0
            for file_ in os.listdir(item):
                if not file_.endswith(".bin"):
                    image_path = os.path.join(item, file_)
                    img = cv2.imread(image_path)
                    detected_face = detect_face_2d(img)
                    if detected_face is not None:
                        bbox = frame_norm(img, detected_face)
                        face_frame = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                        face_landmarks = get_face_landmarks(face_frame)
                        for i in range(5):
                            face_landmarks[i * 2] = face_landmarks[i * 2] + bbox[0]
                            face_landmarks[i * 2 + 1] = face_landmarks[i * 2 + 1] + bbox[1]
                        face_landmarks = face_landmarks.reshape((-1, 2))
                        aligned_face = norm_crop(img, face_landmarks, ref_landmarks)
                        face_features = get_face_features(aligned_face)                            
                        face_features.tofile(item+"/features" + str(count) + ".bin")
                    count += 1
            print("Done processing:", item)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path", default="", type=str,
                        help="Path to the images of the face database (default=%(default)s)")
    args = parser.parse_args()
    generate_database(args.db_path)
