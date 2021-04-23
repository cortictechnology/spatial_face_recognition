import numpy as np
from skimage import transform as trans
import cv2

def estimate_norm(lmk, ref_landmarks):
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float('inf')
    src = ref_landmarks
    
    for i in np.arange(src.shape[0]):
        tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i]) ** 2, axis=1)))
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
    return min_M, min_index

def norm_crop(img, landmark, ref_landmarks):
    M, pose_index = estimate_norm(landmark, ref_landmarks)
    warped = cv2.warpAffine(img, M, (112, 112), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
    return warped

# def to_planar(arr: np.ndarray, shape: tuple) -> list:
def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    resized = cv2.resize(arr, shape, interpolation=cv2.INTER_NEAREST).transpose(2,0,1)
    return resized

def frame_norm(frame, bbox):
        return (np.clip(np.array(bbox), 0, 1) * np.array(frame.shape[:2] * (len(bbox) // 2))[::-1]).astype(int)